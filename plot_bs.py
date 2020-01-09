import matplotlib.pyplot as plt
import pandas as pd
datas = pd.read_csv('Milano_bs.csv')

# print(datas[['lat','lon']].iloc[142])


plt.scatter(x=datas['lon'].astype(float), y=datas['lat'].astype(float))

plt.savefig("results/bs")

plt.clf()