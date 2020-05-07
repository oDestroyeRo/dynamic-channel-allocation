import numpy as np
import matplotlib.pyplot as plt

# kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
traffic_data = np.load("mobile_traffic/npy_merge/merge_traffic.npy")
print(np.max(traffic_data[:, :, :, 1]))
print(np.min(traffic_data[:, :, :, 1]))
print(np.mean(traffic_data[:, :, :, 1]))
print(np.median(traffic_data[:, :, :, 1]))
print(np.std(traffic_data[:, :, :, 1]))

plt.hist(x=traffic_data[0, :, :, 1], y=traffic_data[:, :, :, 1])
# plt.savefig("histrogram.png")

