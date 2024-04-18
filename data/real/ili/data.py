# %%
# %%

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os

from requests import head
# %%
raw_ts = np.load('data/real/ili/sili.npy')

# df = pd.read_csv('data/real/BTC/BTC-USD-6M-2H.csv',header=0,index_col=[0])

# raw_ts = df['close'].values

font = {'family': 'Times New Roman',
        'size': '22'}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(1,1, figsize=(12,4), dpi=300)

ax.set_xlabel('Step', fontsize=22)
ax.set_ylabel('Value', fontsize=22)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(7))
ax.plot(raw_ts, label = 'ILI')

plt.legend(loc='upper left')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()
fig.savefig('data/real/ili/sili.png', dpi= 300, bbox_inches='tight')

plt.close()

# %%
def get_data(df, name):
    data = df[name]
    if data.isnull().any():
        data = data.interpolate()
    
    return data.values
# %%
# np.save(os.path.join(data_folder, 'npsr'), npsr)
# %%
sili = get_data(df, 'south_ILI')
fig=plt.figure(figsize=(10,3))
plt.plot(sili)
plt.show()
# %%

nili = get_data(df, 'north_ILI')
fig=plt.figure(figsize=(10,3))
plt.plot(nili)
plt.show()

# %%
# np.save(os.path.join(data_folder, 'sili'), sili)
# np.save(os.path.join(data_folder, 'nili'), nili)

ts = sili.reshape(-1,)

import statsmodels.api as sm


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_acf(ts, lags=53, ax =ax1)
sm.graphics.tsa.plot_pacf(ts, lags=53, ax=ax2)

# %%
