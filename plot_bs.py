import matplotlib.pyplot as plt
import pandas as pd
datas = pd.read_csv('Milano_TIM_LTE_RBS.csv')


plt.scatter(x=datas['lon'].astype(float), y=datas['lat'].astype(float))

plt.savefig("results/bs")

plt.clf()