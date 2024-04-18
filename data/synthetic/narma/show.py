# %%
import pandas as pd
import numpy as np
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
# %%

ts = np.load('data/synthetic/narma/narma.npy')
print(ts.shape)
folder = 'data/synthetic/narma'
_label = 'NARMA'
file_name = 'narma'

raw_ts = ts
font = {'family': 'Times New Roman',
        'size': '44'}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(1,1, figsize=(32,8), dpi=300)

ax.set_xlabel('Step', fontsize=44)
ax.set_ylabel('Value', fontsize=44)
ax.plot(raw_ts, label = _label)
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))

plt.legend(loc='upper left')
plt.xticks(fontsize = 40)
plt.yticks(fontsize = 40)

plt.show()
fig.savefig(os.path.join(folder, '{}.png'.format(file_name)), dpi= 300, bbox_inches='tight')

plt.close()
# %%
