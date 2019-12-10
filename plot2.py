import matplotlib.pyplot as plt
import pandas as pd


datas = pd.read_csv('results/monitor.csv', names=["reward", "lenght", "", "block_prob"])

datas=datas.astype(float)
# print(datas.head(5))

ax = plt.gca()

datas.plot(kind='line',y='reward',ax=ax)

plt.savefig("results/reward_ppo_multi")

plt.clf()

ax = plt.gca()
datas.plot(kind='line',y='block_prob', color='red', ax=ax)

plt.savefig("results/block_prob_ppo_multi")