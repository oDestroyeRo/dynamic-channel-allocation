import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


datas_ppo= pd.read_csv('results/PPO_test_3.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])
datas_dqn= pd.read_csv('results/DQN_test.csv', names=["reward", "blockprob", "total_blockprob", "Date"])
datas_random = pd.read_csv('results/random_test_3.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])
datas_a2c = pd.read_csv('results/A2C_test_3.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])
datas_dca = pd.read_csv('results/DCA_test_3.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])
# datas_3 = pd.read_csv('results/monitor_channel_16.csv', names=["reward", "lenght", "", "block_prob"])
# datas_4 = pd.read_csv('results/monitor_channel_64.csv', names=["reward", "lenght", "", "block_prob", "date"])

# datas=datas.astype(float)
# datas_2=datas_2.astype(float)
# datas_3=datas_3.astype(float)
# datas_4=datas_4.astype(float)
# print(datas.head(5))

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Compare Model Block Probability')


datas_ppo['blockprob'] = gaussian_filter1d(datas_ppo['blockprob'], sigma=4)
datas_ppo.plot(kind='line',y='blockprob',x='Date',ax=ax, label="PPO")

datas_random['blockprob'] = gaussian_filter1d(datas_random['blockprob'], sigma=4)
datas_random.plot(kind='line',y='blockprob',x='Date',ax=ax, label="Random")

datas_dqn['blockprob'] = gaussian_filter1d(datas_dqn['blockprob'], sigma=4)
datas_dqn.plot(kind='line',y='blockprob',x='Date',ax=ax, label="Deep Q Learning")

datas_a2c['blockprob'] = gaussian_filter1d(datas_a2c['blockprob'], sigma=4)
datas_a2c.plot(kind='line',y='blockprob',x='Date',ax=ax, label="A2C")

datas_dca['blockprob'] = gaussian_filter1d(datas_dca['blockprob'], sigma=4)
datas_dca.plot(kind='line',y='blockprob',x='Date',ax=ax, label="DCA")
# datas_4.plot(kind='line',y='reward',color='yellow',ax=ax, label="64 channels")

plt.savefig("results/blockprob_model")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Compare Block Probability + Drop rate')

datas_ppo['total_blockprob'] = gaussian_filter1d(datas_ppo['total_blockprob'], sigma=4)
datas_ppo.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="PPO")

datas_random['total_blockprob'] = gaussian_filter1d(datas_random['total_blockprob'], sigma=4)
datas_random.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="Random")

datas_dqn['total_blockprob'] = gaussian_filter1d(datas_dqn['total_blockprob'], sigma=4)
datas_dqn.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="Deep Q Learning")

datas_a2c['total_blockprob'] = gaussian_filter1d(datas_a2c['total_blockprob'], sigma=4)
datas_a2c.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="A2C")

datas_dca['total_blockprob'] = gaussian_filter1d(datas_dca['total_blockprob'], sigma=4)
datas_dca.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="DCA")

plt.savefig("results/blockprob_total")
plt.clf()


ax = plt.gca()
ax.set(xlabel='Date', ylabel='Drop rate',
       title='Compare Drop rate')

datas_ppo['drop_rate'] = gaussian_filter1d(datas_ppo['drop_rate'], sigma=4)
datas_ppo.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="PPO")

datas_random['drop_rate'] = gaussian_filter1d(datas_random['drop_rate'], sigma=4)
datas_random.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="Random")

# datas_dqn['drop_rate'] = gaussian_filter1d(datas_dqn['drop_rate'], sigma=4)
# datas_dqn.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="Deep Q Learning")

datas_a2c['drop_rate'] = gaussian_filter1d(datas_a2c['drop_rate'], sigma=4)
datas_a2c.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="A2C")

datas_dca['drop_rate'] = gaussian_filter1d(datas_dca['drop_rate'], sigma=4)
datas_dca.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="DCA")
plt.savefig("results/drop_rate")
plt.clf()

print("Model block")
print("PPO : ", datas_ppo['blockprob'].mean())
print("DQN : ", datas_dqn['blockprob'].mean())
print("RANDOM : ", datas_random['blockprob'].mean())
print("A2C : ", datas_a2c['blockprob'].mean())
print("DCA : ", datas_dca['blockprob'].mean())

print("Total block")
print(datas_ppo['total_blockprob'].mean())
print(datas_dqn['total_blockprob'].mean())
print(datas_random['total_blockprob'].mean())
print(datas_a2c['total_blockprob'].mean())
print(datas_dca['total_blockprob'].mean())

print("Drop rate")
print(datas_ppo['drop_rate'].mean())
# print(datas_dqn['drop_rate'].mean())
print(datas_random['drop_rate'].mean())
print(datas_a2c['drop_rate'].mean())
print(datas_dca['drop_rate'].mean())


