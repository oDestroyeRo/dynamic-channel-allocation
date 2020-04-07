import matplotlib.pyplot as plt
import pandas as pd


datas_ppo= pd.read_csv('results/ppo_real_traffic_15_100m_test_ppo.csv', names=["reward", "blockprob", "total_blockprob", "Date"])
datas_dqn= pd.read_csv('results/ppo_real_traffic_15_100m_test_dqn.csv', names=["reward", "blockprob", "total_blockprob", "Date"])
datas_random = pd.read_csv('results/ppo_real_traffic_15_100m_test_random.csv', names=["reward", "blockprob", "total_blockprob", "Date"])
datas_a2c = pd.read_csv('results/a2c_15_200m_test.csv', names=["reward", "blockprob", "total_blockprob", "Date"])
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

datas_ppo.plot(kind='line',y='blockprob',x='Date',ax=ax, label="PPO")
datas_random.plot(kind='line',y='blockprob',x='Date',ax=ax, label="Random")
datas_dqn.plot(kind='line',y='blockprob',x='Date',ax=ax, label="Deep Q Learning")
datas_a2c.plot(kind='line',y='blockprob',x='Date',ax=ax, label="A2C")
# datas_4.plot(kind='line',y='reward',color='yellow',ax=ax, label="64 channels")

plt.savefig("results/blockprob_model")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Compare Total Block Probability')

datas_ppo.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="PPO")
datas_random.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="Random")
datas_dqn.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="Deep Q Learning")
datas_a2c.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="A2C")

plt.savefig("results/blockprob_total")


print(datas_ppo['blockprob'].mean())
print(datas_dqn['blockprob'].mean())
print(datas_random['blockprob'].mean())
print(datas_a2c['blockprob'].mean())


print(datas_ppo['total_blockprob'].mean())
print(datas_dqn['total_blockprob'].mean())
print(datas_random['total_blockprob'].mean())
print(datas_a2c['total_blockprob'].mean())


