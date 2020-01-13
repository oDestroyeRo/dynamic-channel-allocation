import matplotlib.pyplot as plt
import pandas as pd


datas = pd.read_csv('results/ppo_real_traffic_10_test_2.csv', names=["reward", "block_prob", "date"])
datas_2 = pd.read_csv('results/ppo_real_traffic_10_test_random.csv', names=["reward", "block_prob", "date"])
# datas_3 = pd.read_csv('results/monitor_channel_16.csv', names=["reward", "lenght", "", "block_prob"])
# datas_4 = pd.read_csv('results/monitor_channel_64.csv', names=["reward", "lenght", "", "block_prob", "date"])

# datas=datas.astype(float)
# datas_2=datas_2.astype(float)
# datas_3=datas_3.astype(float)
# datas_4=datas_4.astype(float)
# print(datas.head(5))

ax = plt.gca()

datas.plot(kind='line',y='block_prob',x='date',color='red',ax=ax, label="PPO model")
datas_2.plot(kind='line',y='block_prob',x='date',color='green',ax=ax, label="random")
# datas_4.plot(kind='line',y='reward',color='yellow',ax=ax, label="64 channels")

plt.savefig("results/blockprob_ppo_data")