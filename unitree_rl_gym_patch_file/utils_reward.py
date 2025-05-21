
import numpy as np

# ========= 超参数（与 Isaac Gym reward scale 对齐） ==========
REWARD_CFG = dict(
    tracking_lin_vel = 1.0,
    tracking_ang_vel = 0.5,
    torques = -1e-5,
    lin_vel_z = -2.0,
    ang_vel_xy = -0.05,
    orientation = -1.0,
    dof_vel = -0.1,
    dof_acc = -2.5e-7,
    action_rate = -0.01,
)
TRACKING_SIGMA = 0.25

# ========= 状态缓存 ==========
prev_dof_vel = None
prev_action = None

def reset_reward_state():
    global prev_dof_vel, prev_action
    prev_dof_vel = None
    prev_action = None

def compute_reward_mujoco(d, cmd, tau, qj, dqj, action, default_angles, ang_vel_scale, gravity_orientation):
    global prev_dof_vel, prev_action
    reward = 0.0

    lin_vel = d.qvel[:3]
    ang_vel = d.qvel[3:6]

    lin_vel_error = np.sum((cmd[:2] - lin_vel[:2]) ** 2)
    reward += REWARD_CFG["tracking_lin_vel"] * np.exp(-lin_vel_error / TRACKING_SIGMA)

    ang_vel_error = (cmd[2] - ang_vel[2]) ** 2
    reward += REWARD_CFG["tracking_ang_vel"] * np.exp(-ang_vel_error / TRACKING_SIGMA)

    reward += REWARD_CFG["torques"] * np.sum(tau ** 2)

    reward += REWARD_CFG["lin_vel_z"] * lin_vel[2]**2

    reward += REWARD_CFG["ang_vel_xy"] * np.sum(ang_vel[:2]**2)

    reward += REWARD_CFG["orientation"] * np.sum(gravity_orientation[:2]**2)

    reward += REWARD_CFG["dof_vel"] * np.sum(dqj ** 2)

    if prev_dof_vel is not None:
        acc = (dqj - prev_dof_vel) / d.opt.timestep
        reward += REWARD_CFG["dof_acc"] * np.sum(acc ** 2)
    prev_dof_vel = dqj.copy()

    if prev_action is not None:
        reward += REWARD_CFG["action_rate"] * np.sum((prev_action - action) ** 2)
    prev_action = action.copy()

    return reward
