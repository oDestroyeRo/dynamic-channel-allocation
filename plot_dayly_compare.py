import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


datas_ppo = pd.read_csv('results/PPO/result_mlp.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])

datas_ppo_CNN = pd.read_csv('results/PPO/result_2.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])

blockprob_ppo = {}
total_blockprob_ppo = {}
droprate_ppo = {}
blockprob_ppo_cnn = {}
total_blockprob_ppo_cnn = {}
droprate_ppo_cnn = {}

count = 0
for index, row in datas_ppo.iterrows():
    split_data = row['Date'].split(' ')
    day = split_data[0]
    split_time = split_data[-2].split(':')
    time = split_time[0]
    if (day, time) in blockprob_ppo:
        blockprob_ppo[day, time] += row['blockprob']
        total_blockprob_ppo[day, time] += row['total_blockprob']
        droprate_ppo[day, time] += row['drop_rate']
    else:
        blockprob_ppo[day, time] = row['blockprob']
        total_blockprob_ppo[day, time] = row['total_blockprob']
        droprate_ppo[day, time] = row['drop_rate']
    count += 1

blockprob_ppo[day, time] = blockprob_ppo[day, time]/count
total_blockprob_ppo[day, time] = total_blockprob_ppo[day, time]/count
droprate_ppo[day, time] = droprate_ppo[day, time]/count

print(blockprob_ppo)
print(total_blockprob_ppo)
print(droprate_ppo)
# renew_PPO = 

count = 0
for index, row in datas_ppo_CNN.iterrows():
    split_data = row['Date'].split(' ')
    day = split_data[0]
    split_time = split_data[-2].split(':')
    time = split_time[0]
    if (day, time) in datas_ppo_CNN:
        blockprob_ppo_cnn[day, time] += row['blockprob']
        total_blockprob_ppo_cnn[day, time] += row['total_blockprob']
        droprate_ppo_cnn[day, time] += row['drop_rate']
    else:
        blockprob_ppo_cnn[day, time] = row['blockprob']
        total_blockprob_ppo_cnn[day, time] = row['total_blockprob']
        droprate_ppo_cnn[day, time] = row['drop_rate']
    count += 1

blockprob_ppo_cnn[day, time] = blockprob_ppo_cnn[day, time]/count
total_blockprob_ppo_cnn[day, time] = total_blockprob_ppo_cnn[day, time]/count
droprate_ppo_cnn[day, time] = droprate_ppo_cnn[day, time]/count

print(blockprob_ppo_cnn)
print(total_blockprob_ppo_cnn)
print(droprate_ppo_cnn)



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


