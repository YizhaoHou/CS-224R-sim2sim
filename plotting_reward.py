import numpy as np
import matplotlib.pyplot as plt

reward = np.load("reward_log.npy")
time = np.load("reward_time.npy")

plt.plot(time, reward, label="Reward per step")
plt.xlabel("Time (s)")
plt.ylabel("Reward")
plt.title("Reward vs. Time in MuJoCo rollout")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
