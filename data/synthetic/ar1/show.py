# %%

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
# %%
raw_ts = np.load('data/synthetic/ar1/ar1.npy')

font = {'family': 'Times New Roman',
        'size': '22'}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(1,1, figsize=(12,4), dpi=300)

ax.set_xlabel('Step', fontsize=22)
ax.set_ylabel('Value', fontsize=22)
ax.plot(raw_ts, label = 'AR1')

plt.legend(loc='upper left')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()
fig.savefig('data/synthetic/ar1/ar1.png', dpi= 300, bbox_inches='tight')

plt.close()

# %%
