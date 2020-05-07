import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


datas_ppo = pd.read_csv('results/PPO/result_mlp.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])

datas_ppo_CNN = pd.read_csv('results/PPO/result_2.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])

renew_PPO = {}
for index, row in datas_ppo.iterrows():
    split_data = row['Date'].split(' ')
    day = split_data[0]
    split_time = split_data[-2].split(':')
    time = split_time[0]
    if day=="Sun":
        renew_PPO[day, time] = [row['blockprob'], row['total_blockprob'], row['drop_rate']]
    if day=="Mon":
        renew_PPO[day, time] = [row['blockprob'], row['total_blockprob'], row['drop_rate']]
    if day=="Tue":
        renew_PPO[day, time] = [row['blockprob'], row['total_blockprob'], row['drop_rate']]
    if day=="Wed":
        renew_PPO[day, time] = [row['blockprob'], row['total_blockprob'], row['drop_rate']]
    if day=="Thu":
        renew_PPO[day, time] = [row['blockprob'], row['total_blockprob'], row['drop_rate']]
    if day=="Fri":
        renew_PPO[day, time] = [row['blockprob'], row['total_blockprob'], row['drop_rate']]
    if day=="Sat":
        renew_PPO[day, time] = [row['blockprob'], row['total_blockprob'], row['drop_rate']]

print(renew_PPO)
# renew_PPO = 



ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Compare Block Probability')


datas_ppo['blockprob'] = gaussian_filter1d(datas_ppo['blockprob'], sigma=4)
datas_ppo.plot(kind='line',y='blockprob',x='Date',ax=ax, label="PPO")




datas_ppo_CNN['blockprob'] = gaussian_filter1d(datas_ppo_CNN['blockprob'], sigma=4)
datas_ppo_CNN.plot(kind='line',y='blockprob',x='Date',ax=ax, label="PPO + prediction")
# datas_4.plot(kind='line',y='reward',color='yellow',ax=ax, label="64 channels")

plt.savefig("results/blockprob_model_3")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Compare Total Block Probability')

datas_ppo['total_blockprob'] = gaussian_filter1d(datas_ppo['total_blockprob'], sigma=4)
datas_ppo.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="PPO")



datas_ppo_CNN['total_blockprob'] = gaussian_filter1d(datas_ppo_CNN['total_blockprob'], sigma=4)
datas_ppo_CNN.plot(kind='line',y='total_blockprob',x='Date',ax=ax, label="PPO + prediction")


plt.savefig("results/blockprob_total_3")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Drop rate',
       title='Compare Drop rate')

datas_ppo['drop_rate'] = gaussian_filter1d(datas_ppo['drop_rate'], sigma=4)
datas_ppo.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="PPO")

datas_ppo_CNN['drop_rate'] = gaussian_filter1d(datas_ppo_CNN['drop_rate'], sigma=4)
datas_ppo_CNN.plot(kind='line',y='drop_rate',x='Date',ax=ax, label="PPO + prediction")

plt.savefig("results/droprate_3")

print("Model block")
print(datas_ppo['blockprob'].mean())

print(datas_ppo_CNN['blockprob'].mean())

print("Total block")
print(datas_ppo['total_blockprob'].mean())
print(datas_ppo_CNN['total_blockprob'].mean())

print("Drop rate")
print(datas_ppo['drop_rate'].mean())
print(datas_ppo_CNN['drop_rate'].mean())


