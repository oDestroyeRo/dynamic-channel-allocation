import matplotlib.pyplot as plt
import pandas as pd


datas = pd.read_csv('results/monitor_channel_100_3072step.csv', names=["reward", "lenght", "", "block_prob", "date"])
datas_2 = pd.read_csv('results/monitor_channel_100_random.csv', names=["reward", "lenght", "block_prob", "date"])
# datas_3 = pd.read_csv('results/monitor_channel_16.csv', names=["reward", "lenght", "", "block_prob"])
# datas_4 = pd.read_csv('results/monitor_channel_64.csv', names=["reward", "lenght", "", "block_prob", "date"])

# datas=datas.astype(float)
# datas_2=datas_2.astype(float)
# datas_3=datas_3.astype(float)
# datas_4=datas_4.astype(float)
# print(datas.head(5))

ax = plt.gca()

datas.plot(kind='line',y='reward',color='red',ax=ax, label="PPO model")
datas_2.plot(kind='line',y='reward',color='blue',ax=ax, label="random")
# datas_3.plot(kind='line',y='reward',color='green',ax=ax, label="16 channels")
# datas_4.plot(kind='line',y='reward',color='yellow',ax=ax, label="64 channels")

plt.savefig("results/reward_ppo_multi")

plt.clf()

ax = plt.gca()
datas.plot(kind='line',y='block_prob', color='red', ax=ax, label="PPO model")
datas_2.plot(kind='line',y='block_prob', color='blue', ax=ax, label="random")
# datas_3.plot(kind='line',y='block_prob', color='green', ax=ax, label="16 channels")
# datas_4.plot(kind='line',y='block_prob', color='yellow', ax=ax, label="64 channels")

plt.savefig("results/block_prob_ppo_multi")