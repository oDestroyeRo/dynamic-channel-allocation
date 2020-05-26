import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


datas_ppo = pd.read_csv('results/PPO/result_mlp.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])

datas_ppo_CNN = pd.read_csv('results/PPO/result_2.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])

datas_DCA = pd.read_csv('results/DCA/result_3.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])

datas_A2C = pd.read_csv('results/A2C/result_3.csv', names=["reward", "blockprob", "total_blockprob", "drop_rate", "Date"])





blockprob_ppo = {}
total_blockprob_ppo = {}
droprate_ppo = {}
blockprob_ppo_cnn = {}
total_blockprob_ppo_cnn = {}
droprate_ppo_cnn = {}
blockprob_dca = {}
total_blockprob_dca = {}
droprate_dca= {}

blockprob_a2c = {}
total_blockprob_a2c = {}
droprate_a2c= {}

count = 0
for index, row in datas_ppo.iterrows():
    split_data = row['Date'].split(' ')
    day = split_data[0]
    split_time = split_data[-2].split(':')
    time = split_time[0]
    if (day, time) in datas_ppo:
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


count = 0
for index, row in datas_DCA.iterrows():
    split_data = row['Date'].split(' ')
    day = split_data[0]
    split_time = split_data[-2].split(':')
    time = split_time[0]
    if (day, time) in datas_DCA:
        blockprob_dca[day, time] += row['blockprob']
        total_blockprob_dca[day, time] += row['total_blockprob']
        droprate_dca[day, time] += row['drop_rate']
    else:
        blockprob_dca[day, time] = row['blockprob']
        total_blockprob_dca[day, time] = row['total_blockprob']
        droprate_dca[day, time] = row['drop_rate']
    count += 1
blockprob_dca[day, time] = blockprob_dca[day, time]/count
total_blockprob_dca[day, time] = total_blockprob_dca[day, time]/count
droprate_dca[day, time] = droprate_dca[day, time]/count

count = 0
for index, row in datas_A2C.iterrows():
    split_data = row['Date'].split(' ')
    day = split_data[0]
    split_time = split_data[-2].split(':')
    time = split_time[0]
    if (day, time) in datas_A2C:
        blockprob_a2c[day, time] += row['blockprob']
        total_blockprob_a2c[day, time] += row['total_blockprob']
        droprate_a2c[day, time] += row['drop_rate']
    else:
        blockprob_a2c[day, time] = row['blockprob']
        total_blockprob_a2c[day, time] = row['total_blockprob']
        droprate_a2c[day, time] = row['drop_rate']
    count += 1
blockprob_a2c[day, time] = blockprob_a2c[day, time]/count
total_blockprob_a2c[day, time] = total_blockprob_a2c[day, time]/count
droprate_a2c[day, time] = droprate_a2c[day, time]/count


# print(blockprob_ppo_cnn)
# print(total_blockprob_ppo_cnn)
# print(droprate_ppo_cnn)
dic_ppo = {}
array_ppo = []
for key, key2 in blockprob_ppo:
    dic_ppo = {}
    dic_ppo['day'] = key
    dic_ppo['time'] = key2
    dic_ppo['blockprob'] = blockprob_ppo[key, key2]
    dic_ppo['total_blcokprob'] = total_blockprob_ppo[key, key2]
    dic_ppo['droprate'] = droprate_ppo[key, key2]
    array_ppo.append(dic_ppo)



dic_ppo_cnn = {}
array_ppo_cnn = []
for key, key2 in blockprob_ppo_cnn:
    dic_ppo_cnn = {}
    dic_ppo_cnn['day'] = key
    dic_ppo_cnn['time'] = key2
    dic_ppo_cnn['blockprob'] = blockprob_ppo_cnn[key, key2]
    dic_ppo_cnn['total_blockprob'] = total_blockprob_ppo_cnn[key, key2]
    dic_ppo_cnn['droprate'] = droprate_ppo_cnn[key, key2]

    array_ppo_cnn.append(dic_ppo_cnn)


dic_dca = {}
array_dca = []
for key, key2 in blockprob_dca:
    dic_dca = {}
    dic_dca['day'] = key
    dic_dca['time'] = key2
    dic_dca['blockprob'] = blockprob_dca[key, key2]
    dic_dca['total_blcokprob'] = total_blockprob_dca[key, key2]
    dic_dca['droprate'] = droprate_dca[key, key2]
    array_dca.append(dic_dca)

dic_a2c = {}
array_a2c = []
for key, key2 in blockprob_a2c:
    dic_a2c = {}
    dic_a2c['day'] = key
    dic_a2c['time'] = key2
    dic_a2c['blockprob'] = blockprob_a2c[key, key2]
    dic_a2c['total_blcokprob'] = total_blockprob_a2c[key, key2]
    dic_a2c['droprate'] = droprate_a2c[key, key2]
    array_a2c.append(dic_a2c)


dic_ppo = pd.DataFrame.from_dict(array_ppo)
dic_ppo_cnn = pd.DataFrame.from_dict(array_ppo_cnn)
dic_dca = pd.DataFrame.from_dict(array_dca)
dic_a2c = pd.DataFrame.from_dict(array_a2c)


# print(dic_ppo)
# print(dic_ppo_cnn)
ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Average Block Probability for Daily')

# dic_ppo['blockprob'] = gaussian_filter1d(dic_ppo['blockprob'], sigma=4)
# dic_ppo.plot(kind='line',y='blockprob',x='time',ax=ax, label="A2C")




dic_ppo_cnn['blockprob'] = gaussian_filter1d(dic_ppo_cnn['blockprob'], sigma=2)
sun_ppo_cnn = dic_ppo_cnn['day'] == "Sun"
mon_ppo_cnn = dic_ppo_cnn['day'] == "Mon"
tue_ppo_cnn = dic_ppo_cnn['day'] == "Tue"
wed_ppo_cnn = dic_ppo_cnn['day'] == "Wed"
thu_ppo_cnn = dic_ppo_cnn['day'] == "Thu"
fri_ppo_cnn = dic_ppo_cnn['day'] == "Fri"
sat_ppo_cnn = dic_ppo_cnn['day'] == "Sat"

dic_ppo_cnn[sun_ppo_cnn].plot(kind='line',y='blockprob',x='time',ax=ax, label="Sunday", marker="o", linestyle='--', markevery=5)
dic_ppo_cnn[mon_ppo_cnn].plot(kind='line',y='blockprob',x='time',ax=ax, label="Monday", marker="p", markevery=5)
dic_ppo_cnn[tue_ppo_cnn].plot(kind='line',y='blockprob',x='time',ax=ax, label="Tuesday", marker="s", linestyle='dotted', markevery=5)
dic_ppo_cnn[wed_ppo_cnn].plot(kind='line',y='blockprob',x='time',ax=ax, label="Wednesday", marker="H", linestyle='--', markevery=5)
dic_ppo_cnn[thu_ppo_cnn].plot(kind='line',y='blockprob',x='time',ax=ax, label="Thursday", marker="d", markevery=5)
dic_ppo_cnn[fri_ppo_cnn].plot(kind='line',y='blockprob',x='time',ax=ax, label="Friday", marker="^", linestyle='dotted', markevery=5)
dic_ppo_cnn[sat_ppo_cnn].plot(kind='line',y='blockprob',x='time',ax=ax, label="Saturday", marker="x", linestyle='--', markevery=5)
ax.legend()
# dic_ppo_cnn.plot(kind='line',y='blockprob',x='time',ax=ax, label="PPO + prediction")
# datas_4.plot(kind='line',y='reward',color='yellow',ax=ax, label="64 channels")


plt.savefig("results/blockprob_model_3.svg")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Block Probability',
       title='Average Block probability + Droprate for Daily')
dic_ppo_cnn['total_blockprob'] = gaussian_filter1d(dic_ppo_cnn['total_blockprob'], sigma=2)

dic_ppo_cnn[sun_ppo_cnn].plot(kind='line',y='total_blockprob',x='time',ax=ax, label="Sunday", marker="o", linestyle='--', markevery=5)
dic_ppo_cnn[mon_ppo_cnn].plot(kind='line',y='total_blockprob',x='time',ax=ax, label="Monday", marker="p", markevery=5)
dic_ppo_cnn[tue_ppo_cnn].plot(kind='line',y='total_blockprob',x='time',ax=ax, label="Tuesday", marker="s", linestyle='dotted', markevery=5)
dic_ppo_cnn[wed_ppo_cnn].plot(kind='line',y='total_blockprob',x='time',ax=ax, label="Wednesday", marker="H", linestyle='--', markevery=5)
dic_ppo_cnn[thu_ppo_cnn].plot(kind='line',y='total_blockprob',x='time',ax=ax, label="Thursday", marker="d", markevery=5)
dic_ppo_cnn[fri_ppo_cnn].plot(kind='line',y='total_blockprob',x='time',ax=ax, label="Friday", marker="^", linestyle='dotted', markevery=5)
dic_ppo_cnn[sat_ppo_cnn].plot(kind='line',y='total_blockprob',x='time',ax=ax, label="Saturday", marker="x", linestyle='--', markevery=5)
ax.legend()

plt.savefig("results/blockprob_total_3.svg")
plt.clf()

ax = plt.gca()
ax.set(xlabel='Date', ylabel='Drop rate',
       title='Average  Droprate for Daily')
dic_ppo_cnn['droprate'] = gaussian_filter1d(dic_ppo_cnn['droprate'], sigma=2)

dic_ppo_cnn[sun_ppo_cnn].plot(kind='line',y='droprate',x='time',ax=ax, label="Sunday", marker="o", linestyle='--', markevery=5)
dic_ppo_cnn[mon_ppo_cnn].plot(kind='line',y='droprate',x='time',ax=ax, label="Monday", marker="p", markevery=5)
dic_ppo_cnn[tue_ppo_cnn].plot(kind='line',y='droprate',x='time',ax=ax, label="Tuesday", marker="s", linestyle='dotted', markevery=5)
dic_ppo_cnn[wed_ppo_cnn].plot(kind='line',y='droprate',x='time',ax=ax, label="Wednesday", marker="H", linestyle='--', markevery=5)
dic_ppo_cnn[thu_ppo_cnn].plot(kind='line',y='droprate',x='time',ax=ax, label="Thursday", marker="d", markevery=5)
dic_ppo_cnn[fri_ppo_cnn].plot(kind='line',y='droprate',x='time',ax=ax, label="Friday", marker="^", linestyle='dotted', markevery=5)
dic_ppo_cnn[sat_ppo_cnn].plot(kind='line',y='droprate',x='time',ax=ax, label="Saturday", marker="x", linestyle='--', markevery=5)
ax.legend()

plt.savefig("results/droprate_3.png")
plt.clf()
# print("Model block")
# print(datas_ppo['blockprob'].mean())

# print(datas_ppo_CNN['blockprob'].mean())

# print("Total block")
# print(datas_ppo['total_blockprob'].mean())
# print(datas_ppo_CNN['total_blockprob'].mean())

# print("Drop rate")
# print(datas_ppo['drop_rate'].mean())
# print(datas_ppo_CNN['drop_rate'].mean())


# print(dic_ppo_cnn[sun_ppo_cnn].mean())
# print(dic_ppo_cnn[mon_ppo_cnn].mean())
# print(dic_ppo_cnn[tue_ppo_cnn].mean())
# print(dic_ppo_cnn[wed_ppo_cnn].mean())
# print(dic_ppo_cnn[thu_ppo_cnn].mean())
# print(dic_ppo_cnn[fri_ppo_cnn].mean())
# print(dic_ppo_cnn[sat_ppo_cnn].mean())


ax = plt.gca()
ax.set(xlabel='Date', ylabel='Blockprob',
       title='Compare average  block probability for Daily')
dic_ppo_cnn.groupby("day").mean().plot(kind='bar', y='blockprob', label="PPO + prediction", ax=ax, color = 'C0', width = 0.15, position=1)
dic_ppo.groupby("day").mean().plot(kind='bar', y='blockprob', label="PPO", ax=ax, color = 'C1', width = 0.15, position=2)
dic_dca.groupby("day").mean().plot(kind='bar', y='blockprob', label="DCA", ax=ax, color = 'C2', width = 0.15, position=3)
dic_a2c.groupby("day").mean().plot(kind='bar', y='blockprob', label="A2C", ax=ax, color = 'C3', width = 0.15, position=4)

print(dic_a2c.head())
print(dic_ppo.head())
ax.legend()
plt.savefig("results/blockprob_4.svg")
# dic_ppo_cnn[sat_ppo_cnn].mean().plot(kind='bar', y='blockprob')