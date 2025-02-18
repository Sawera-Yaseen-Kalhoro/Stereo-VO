import numpy as np
import matplotlib.pyplot as plt

est_data = np.loadtxt("evo_eval/results/10_est.txt")
gt_data = np.loadtxt("evo_eval/results/10_gt.txt")

est_trajectory = est_data[:, [3, 11]]
gt_trajectory = gt_data[:, [3, 11]]

plt.figure()

plt.plot(est_trajectory[:, 0], est_trajectory[:, 1], label='Estimated Trajectory', color='blue')

plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label='Ground Truth Trajectory', color='green')

plt.title("Trajectories (X vs Z)")
plt.xlabel("X")
plt.ylabel("Z")
plt.legend()

plt.show()
