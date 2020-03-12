import matplotlib.pyplot as plt
import pandas as pd
datas = pd.read_csv('Reuse_factor.csv', names=["factor", "block_prob"])

ax = plt.gca()

datas.plot(kind='line',y='block_prob',x='factor',color='red',ax=ax)
# datas_4.plot(kind='line',y='reward',color='yellow',ax=ax, label="64 channels")

plt.savefig("results/reuse_factor")