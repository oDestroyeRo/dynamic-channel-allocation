import matplotlib.pyplot as plt
import pandas as pd


datas = pd.read_csv('results/dqn_init_70_ran.csv', names=["step", "block_prob", "reward", ""])
# print(datas.head(5))

ax = plt.gca()

datas.plot(kind='line',x='step',y='block_prob',ax=ax)

plt.savefig("results/block_prob")

plt.clf()

ax = plt.gca()
datas.plot(kind='line',x='step',y='reward', color='red', ax=ax)

plt.savefig("results/reward")


# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numbers')
# plt.show()