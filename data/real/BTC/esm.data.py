# %%

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os

from requests import head
# %%
# raw_ts = np.load('data/synthetic/ar1/ar1.npy')

df = pd.read_csv('data/real/BTC/BTC-USD-6M-2H.csv',header=0,index_col=[0])

raw_ts = df['close'].values

font = {'family': 'Times New Roman',
        'size': '22'}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(1,1, figsize=(12,4), dpi=300)

ax.set_xlabel('Step', fontsize=22)
ax.set_ylabel('Value', fontsize=22)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
ax.plot(raw_ts, label = 'BTC')

plt.legend(loc='upper left')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()
fig.savefig('data/real/BTC/btc.png', dpi= 300, bbox_inches='tight')

plt.close()

# %%
