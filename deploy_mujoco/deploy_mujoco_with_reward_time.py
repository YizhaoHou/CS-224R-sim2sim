import time
import os
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import argparse
from utils_reward import compute_reward_mujoco, reset_reward_state

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz)
    ])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument("--headless", action="store_true", help="Run without rendering viewer")
    args = parser.parse_args()

    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)

    reset_reward_state()
    reward_log = []
    reward_timeline = []
    total_reward = 0.0

    def simulate_step():
        global counter, action, target_dof_pos, total_reward
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        qj = d.qpos[7:]
        dqj = d.qvel[6:]
        quat = d.qpos[3:7]
        omega = d.qvel[3:6]
        base_height = d.qpos[2]

        qj_scaled = (qj - default_angles) * dof_pos_scale
        dqj_scaled = dqj * dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * ang_vel_scale

        r_step = compute_reward_mujoco(
            d=d,
            cmd=cmd,
            tau=tau,
            qj=qj_scaled,
            dqj=dqj_scaled,
            action=action,
            default_angles=default_angles,
            ang_vel_scale=ang_vel_scale,
            gravity_orientation=gravity_orientation,
            timestep=simulation_dt
        )
        reward_log.append(r_step)
        reward_timeline.append(counter * simulation_dt)
        total_reward += r_step

        counter += 1
        if counter % control_decimation == 0:
            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega_scaled
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + num_actions] = qj_scaled
            obs[9 + num_actions:9 + 2 * num_actions] = dqj_scaled
            obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            obs[47] = base_height  # ensure full 48D observation

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

    # ================================
    # 🧠 Main Loop (GUI or Headless)
    # ================================
    start = time.time()

    if args.headless:
        num_steps = int(simulation_duration / simulation_dt)
        for _ in range(num_steps):
            step_start = time.time()
            simulate_step()
            time.sleep(max(0, m.opt.timestep - (time.time() - step_start)))
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and time.time() - start < simulation_duration:
                step_start = time.time()
                simulate_step()
                viewer.sync()
                time.sleep(max(0, m.opt.timestep - (time.time() - step_start)))

    # ================================
    # ✅ Save Rewards & Print Summary
    # ================================
    reward_log = np.array(reward_log)
    reward_timeline = np.array(reward_timeline)
    np.save("reward_log.npy", reward_log)
    np.save("reward_time.npy", reward_timeline)
    print(f"\n✅ Total reward over rollout: {total_reward:.3f}")
    print("📁 Saved: reward_log.npy and reward_time.npy")
