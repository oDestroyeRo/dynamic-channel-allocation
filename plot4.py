import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


datas_ppo= pd.read_csv('results/PPO/result_mlp.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])
# datas_dca = pd.read_csv('results/DCA_test.csv', names=["reward", "blockprob", "total_blockprob", "Date"])
datas_ppo_CNN = pd.read_csv('results/PPO/result_2.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])
# datas_3 = pd.read_csv('results/monitor_channel_16.csv', names=["reward", "lenght", "", "block_prob"])
# datas_4 = pd.read_csv('results/monitor_channel_64.csv', names=["reward", "lenght", "", "block_prob", "date"])
datas_a2c= pd.read_csv('results/A2C/result_3.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])

datas_dca = pd.read_csv('results/DCA/result_3.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])
# datas=datas.astype(float)
# datas_2=datas_2.astype(float)
# datas_3=datas_3.astype(float)
# datas_4=datas_4.astype(float)
# print(datas.head(5))

datas_ppo = datas_ppo.drop(datas_ppo.index[0:7664])
datas_ppo_CNN = datas_ppo_CNN.drop(datas_ppo_CNN.index[0:7664])
datas_a2c = datas_a2c.drop(datas_a2c.index[0:7664])
datas_dca = datas_dca.drop(datas_dca.index[0:7664])

datas_ppo["Date"] = datas_ppo["Date"].str.extract(pat = '([A-Z][a-z][a-z] [A-Z][a-z][a-z] [0-9][0-9])') 
datas_ppo_CNN["Date"] = datas_ppo_CNN["Date"].str.extract(pat = '([A-Z][a-z][a-z] [A-Z][a-z][a-z] [0-9][0-9])')
datas_a2c["Date"] = datas_a2c["Date"].str.extract(pat = '([A-Z][a-z][a-z] [A-Z][a-z][a-z] [0-9][0-9])')
datas_dca["Date"] = datas_dca["Date"].str.extract(pat = '([A-Z][a-z][a-z] [A-Z][a-z][a-z] [0-9][0-9])')



ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Compare Block Probability')

datas_ppo['blockprob'] = gaussian_filter1d(datas_ppo['blockprob'], sigma=4)
datas_ppo.plot(kind='line',y='blockprob',x='Date',ax=ax, label="PPO", marker="o", linestyle='--', markevery=30)


# datas_dca['blockprob'] = gaussian_filter1d(datas_dca['blockprob'], sigma=4)
# datas_dca.plot(kind='line',y='blockprob',x='Date',ax=ax, label="DCA")

datas_ppo_CNN['blockprob'] = gaussian_filter1d(datas_ppo_CNN['blockprob'], sigma=4)
datas_ppo_CNN.plot(kind='line',y='blockprob',x='Date',ax=ax, label="PPO + prediction", marker="p", markevery=30)

datas_a2c['blockprob'] = gaussian_filter1d(datas_a2c['blockprob'], sigma=4)
datas_a2c.plot(kind='line',y='blockprob',x='Date',ax=ax, label="A2C", marker="s", linestyle='--', markevery=30)

datas_dca['blockprob'] = gaussian_filter1d(datas_dca['blockprob'], sigma=4)
datas_dca.plot(kind='line',y='blockprob',x='Date',ax=ax, label="DCA", marker="x", linestyle='dotted', markevery=30)
# datas_4.plot(kind='line',y='reward',color='yellow',ax=ax, label="64 channels")
ax.legend()
plt.savefig("results/blockprob_model_2.svg")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Compare Total Block Probability')
datas_ppo['total_blockprob'] = gaussian_filter1d(datas_ppo['total_blockprob'], sigma=4)
datas_ppo.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="PPO", marker="o", linestyle='--', markevery=30)

# datas_dca['total_blockprob'] = gaussian_filter1d(datas_dca['total_blockprob'], sigma=4)
# datas_dca.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="DCA")

datas_ppo_CNN['total_blockprob'] = gaussian_filter1d(datas_ppo_CNN['total_blockprob'], sigma=4)
datas_ppo_CNN.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="PPO + prediction", marker="p", markevery=30)

datas_a2c['total_blockprob'] = gaussian_filter1d(datas_a2c['total_blockprob'], sigma=4)
datas_a2c.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="A2C", marker="s", linestyle='--', markevery=30)

datas_dca['total_blockprob'] = gaussian_filter1d(datas_dca['total_blockprob'], sigma=4)
datas_dca.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="DCA", linestyle='dotted', marker="x", markevery=30)

ax.legend()
plt.savefig("results/blockprob_total_2.svg")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Drop rate',
       title='Compare Drop rate')
datas_ppo['drop_rate'] = gaussian_filter1d(datas_ppo['drop_rate'], sigma=4)
datas_ppo.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="PPO", marker="o", linestyle='--', markevery=30)

datas_ppo_CNN['drop_rate'] = gaussian_filter1d(datas_ppo_CNN['drop_rate'], sigma=4)
datas_ppo_CNN.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="PPO + prediction", marker="p", markevery=30)

datas_a2c['drop_rate'] = gaussian_filter1d(datas_a2c['drop_rate'], sigma=4)
datas_a2c.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="A2C", marker="s", linestyle='--', markevery=30)

datas_dca['drop_rate'] = gaussian_filter1d(datas_dca['drop_rate'], sigma=4)
datas_dca.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="DCA", linestyle='dotted', marker="x", markevery=30)
ax.legend()
plt.savefig("results/droprate_2.svg")

print("Model block")
print(datas_ppo['blockprob'].mean())

print(datas_ppo_CNN['blockprob'].mean())

print("Total block")
print(datas_ppo['total_blockprob'].mean())
print(datas_ppo_CNN['total_blockprob'].mean())

print("Drop rate")
print(datas_ppo['drop_rate'].mean())
print(datas_ppo_CNN['drop_rate'].mean())


