import matplotlib.pyplot as plt
import pandas as pd
datas = pd.read_csv('Milano_TIM_LTE_RBS.csv')
# print(datas.head(5))

# ax = plt.gca()

# samples = datas['samples'] == 1

# lat = datas['lat']

# lon = datas['lon']

# datas.lat=pd.to_numeric(datas.lat)
# datas.lon=pd.to_numeric(datas.lon)
# print(datas['lat'].min())

# print(datas['lon')

# lat = datas['lat'] > 50

# print(datas['lon'] < 9.2)

plt.scatter(x=datas['lon'].astype(float), y=datas['lat'].astype(float))

# plt.xlim((9.0,10))
# plt.ylim((45.3,46))

plt.savefig("results/bs")

plt.clf()