import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

traffic_data = np.load("mobile_traffic/npy_merge/merge_traffic.npy")
print(np.min(traffic_data[0:144,:,:,1]), np.max(traffic_data[0:144,:,:,1]), np.sum(traffic_data[0:144,:,:,1]))
new_traffic = np.zeros(144)
count = 0
for i in range(12):
    for j in range(12):
        new_traffic[count] = traffic_data[0,i,j,1]
        count += 1

# print(new_traffic)
print(np.sum(new_traffic)/10)
plt.hist(new_traffic)
plt.savefig("results/traffic")