import matplotlib.pyplot as plt
import pandas as pd


datas = pd.read_csv('tmp/monitor.csv', names=["reward", "lenght", ""])

datas=datas.astype(float)
# print(datas.head(5))

ax = plt.gca()

datas.plot(kind='line',y='reward',ax=ax)

plt.savefig("tmp/reward_ppo")

plt.clf()

# ax = plt.gca()
# datas.plot(kind='line',y='reward', color='red', ax=ax)

# plt.savefig("results/reward_multi")