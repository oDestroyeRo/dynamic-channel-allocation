import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


datas_ppo= pd.read_csv('results/PPO_test_3.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])
# datas_dca = pd.read_csv('results/DCA_test.csv', names=["reward", "blockprob", "total_blockprob", "Date"])
datas_ppo_CNN = pd.read_csv('results/PPO/result.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])
# datas_3 = pd.read_csv('results/monitor_channel_16.csv', names=["reward", "lenght", "", "block_prob"])
# datas_4 = pd.read_csv('results/monitor_channel_64.csv', names=["reward", "lenght", "", "block_prob", "date"])

# datas=datas.astype(float)
# datas_2=datas_2.astype(float)
# datas_3=datas_3.astype(float)
# datas_4=datas_4.astype(float)
# print(datas.head(5))


ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Compare Block Probability')


datas_ppo['blockprob'] = gaussian_filter1d(datas_ppo['blockprob'], sigma=4)
datas_ppo.plot(kind='line',y='blockprob',x='Date',ax=ax, label="PPO")


# datas_dca['blockprob'] = gaussian_filter1d(datas_dca['blockprob'], sigma=4)
# datas_dca.plot(kind='line',y='blockprob',x='Date',ax=ax, label="DCA")

datas_ppo_CNN['blockprob'] = gaussian_filter1d(datas_ppo_CNN['blockprob'], sigma=4)
datas_ppo_CNN.plot(kind='line',y='blockprob',x='Date',ax=ax, label="PPO + prediction")
# datas_4.plot(kind='line',y='reward',color='yellow',ax=ax, label="64 channels")

plt.savefig("results/blockprob_model_2")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Compare Total Block Probability')

datas_ppo['total_blockprob'] = gaussian_filter1d(datas_ppo['total_blockprob'], sigma=4)
datas_ppo.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="PPO")

# datas_dca['total_blockprob'] = gaussian_filter1d(datas_dca['total_blockprob'], sigma=4)
# datas_dca.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="DCA")

datas_ppo_CNN['total_blockprob'] = gaussian_filter1d(datas_ppo_CNN['total_blockprob'], sigma=4)
datas_ppo_CNN.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="PPO + prediction")


plt.savefig("results/blockprob_total_2")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Drop rate',
       title='Compare Drop rate')

datas_ppo['drop_rate'] = gaussian_filter1d(datas_ppo['drop_rate'], sigma=4)
datas_ppo.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="PPO")

datas_ppo_CNN['drop_rate'] = gaussian_filter1d(datas_ppo_CNN['drop_rate'], sigma=4)
datas_ppo_CNN.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="PPO + prediction")

plt.savefig("results/droprate_2")

print("Model block")
print(datas_ppo['blockprob'].mean())

print(datas_ppo_CNN['blockprob'].mean())

print("Total block")
print(datas_ppo['total_blockprob'].mean())
print(datas_ppo_CNN['total_blockprob'].mean())

print("Drop rate")
print(datas_ppo['drop_rate'].mean())
print(datas_ppo_CNN['drop_rate'].mean())


