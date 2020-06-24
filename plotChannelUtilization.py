import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


datas_s1 = pd.read_csv('results/PPO/result_sin_1_500.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s2 = pd.read_csv('results/PPO/result_sin_2_510.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s3 = pd.read_csv('results/PPO/result_sin_3_520.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s4 = pd.read_csv('results/PPO/result_sin_4_490.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s5 = pd.read_csv('results/PPO/result_sin_5_480.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s6 = pd.read_csv('results/PPO/result_sin_6_470.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s7 = pd.read_csv('results/PPO/result_sin_7_460.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s8 = pd.read_csv('results/PPO/result_sin_8_530.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s9 = pd.read_csv('results/PPO/result_sin_9_540.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s10 = pd.read_csv('results/PPO/result_sin_10_550.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s11 = pd.read_csv('results/PPO/result_sin_12_505.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
datas_s12 = pd.read_csv('results/PPO/result_sin_11_560.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
# datas_s13 = pd.read_csv('results/PPO/result_sin_13_515.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
# datas_s14  = pd.read_csv('results/PPO/result_sin_14_525.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
# datas_s15 = pd.read_csv('results/PPO/result_sin_15_535.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])
# datas_s16 = pd.read_csv('results/PPO/result_sin_16_495.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date", "Util"])

# name_util = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
# dotas_util = [1-datas_s1["Util"].mean(), 1-datas_s2["Util"].mean(), 1-datas_s3["Util"].mean(), 1-datas_s4["Util"].mean(), 1-datas_s5["Util"].mean(), 1-datas_s6["Util"].mean(), 1-datas_s7["Util"].mean(), 1-datas_s8["Util"].mean()
# , 1-datas_s9["Util"].mean(), 1-datas_s10["Util"].mean(), 1-datas_s11["Util"].mean(), 1-datas_s12["Util"].mean(), 1-datas_s13["Util"].mean(), 1-datas_s14["Util"].mean(), 1-datas_s15["Util"].mean(), 1-datas_s6["Util"].mean()]
name_util = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]
dotas_util = [1-datas_s1["Util"].mean(), 1-datas_s2["Util"].mean(), 1-datas_s3["Util"].mean(), 1-datas_s4["Util"].mean(), 1-datas_s5["Util"].mean(), 1-datas_s6["Util"].mean(), 1-datas_s7["Util"].mean(), 1-datas_s8["Util"].mean()
, 1-datas_s9["Util"].mean(), 1-datas_s10["Util"].mean(), 1-datas_s11["Util"].mean(), 1-datas_s12["Util"].mean()]

ax = plt.gca()
ax.set(xlabel='Senario', ylabel='Channel Utilization',
       title='Compare Channel Utilization')

bars = plt.bar(name_util, dotas_util)
# for i, v in enumerate(dotas_util):
#        ax.text(v, i - .25, str(round(v,2)), color='blue', fontweight='bold')
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x(), yval + .005, str(round(yval,2)))

# ax.legend()
plt.savefig("results/Util.svg")
plt.clf()

name_util = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]
dotas_util = [datas_s1["blockprob"].mean(), datas_s2["blockprob"].mean(), datas_s3["blockprob"].mean(), datas_s4["blockprob"].mean(), datas_s5["blockprob"].mean(), datas_s6["blockprob"].mean(), datas_s7["blockprob"].mean(), datas_s8["blockprob"].mean()
, datas_s9["blockprob"].mean(), datas_s10["blockprob"].mean(), datas_s11["blockprob"].mean(), datas_s12["blockprob"].mean()]

ax = plt.gca()
ax.set(xlabel='Senario', ylabel='Block Probability',
       title='Compare Block Probability')

bars = plt.bar(name_util, dotas_util)
# for i, v in enumerate(dotas_util):
#        ax.text(v, i - .25, str(round(v,2)), color='blue', fontweight='bold')
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x(), yval + .005, str(round(yval,2)))

# ax.legend()
plt.savefig("results/BlockProb.svg")
plt.clf()

name_util = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]
dotas_util = [datas_s1["total_blockprob"].mean(), datas_s2["total_blockprob"].mean(), datas_s3["total_blockprob"].mean(), datas_s4["total_blockprob"].mean(), datas_s5["total_blockprob"].mean(), datas_s6["total_blockprob"].mean(), datas_s7["total_blockprob"].mean(), datas_s8["total_blockprob"].mean()
, datas_s9["total_blockprob"].mean(), datas_s10["total_blockprob"].mean(), datas_s11["total_blockprob"].mean(), datas_s12["total_blockprob"].mean()]

ax = plt.gca()
ax.set(xlabel='Senario', ylabel='Block Probability',
       title='Compare Total Block Probability')

bars = plt.bar(name_util, dotas_util)
# for i, v in enumerate(dotas_util):
#        ax.text(v, i - .25, str(round(v,2)), color='blue', fontweight='bold')
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x(), yval + .005, str(round(yval,2)))

# ax.legend()
plt.savefig("results/Total_BlockProb.svg")
plt.clf()

print("Model block")
print(datas_s1['blockprob'].mean())
print(datas_s2['blockprob'].mean())
print(datas_s3['blockprob'].mean())
print(datas_s4['blockprob'].mean())
print(datas_s5['blockprob'].mean())
print(datas_s6['blockprob'].mean())
print(datas_s7['blockprob'].mean())
print(datas_s8['blockprob'].mean())
print(datas_s9['blockprob'].mean())
print(datas_s10['blockprob'].mean())
print(datas_s11['blockprob'].mean())
print(datas_s12['blockprob'].mean())
# print(datas_s13['blockprob'].mean())
# print(datas_s14['blockprob'].mean())
# print(datas_s15['blockprob'].mean())
# print(datas_s16['blockprob'].mean())

print("Model Util")
print(datas_s1['Util'].mean())
print(datas_s2['Util'].mean())
print(datas_s3['Util'].mean())
print(datas_s4['Util'].mean())
print(datas_s5['Util'].mean())
print(datas_s6['Util'].mean())
print(datas_s7['Util'].mean())
print(datas_s8['Util'].mean())
print(datas_s9['Util'].mean())
print(datas_s10['Util'].mean())
print(datas_s11['Util'].mean())
print(datas_s12['Util'].mean())
# print(1-datas_s13['Util'].mean())
# print(1-datas_s14['Util'].mean())
# print(1-datas_s15['Util'].mean())
# print(1-datas_s16['Util'].mean())


print("Total block")
print(datas_s1['total_blockprob'].mean())
print(datas_s2['total_blockprob'].mean())
print(datas_s3['total_blockprob'].mean())
print(datas_s4['total_blockprob'].mean())
print(datas_s5['total_blockprob'].mean())
print(datas_s6['total_blockprob'].mean())
print(datas_s7['total_blockprob'].mean())
print(datas_s8['total_blockprob'].mean())
print(datas_s9['total_blockprob'].mean())
print(datas_s10['total_blockprob'].mean())
print(datas_s11['total_blockprob'].mean())
print(datas_s12['total_blockprob'].mean())
# print(datas_s13['total_blockprob'].mean())
# print(datas_s14['total_blockprob'].mean())
# print(datas_s15['total_blockprob'].mean())
# print(datas_s16['total_blockprob'].mean())

print("Drop rate")
print(datas_s1['drop_rate'].mean())
print(datas_s2['drop_rate'].mean())
print(datas_s3['drop_rate'].mean())
print(datas_s4['drop_rate'].mean())
print(datas_s5['drop_rate'].mean())
print(datas_s6['drop_rate'].mean())
print(datas_s7['drop_rate'].mean())
print(datas_s8['drop_rate'].mean())
print(datas_s9['drop_rate'].mean())
print(datas_s10['drop_rate'].mean())
print(datas_s11['drop_rate'].mean())
print(datas_s12['drop_rate'].mean())
# print(datas_s13['drop_rate'].mean())
# print(datas_s14['drop_rate'].mean())
# print(datas_s15['drop_rate'].mean())
# print(datas_s16['drop_rate'].mean())