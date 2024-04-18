# %%
import pandas as pd
import numpy as np
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
# %%

ts = np.load('data/synthetic/mg/mg.npy')
print(ts.shape)
folder = 'data/synthetic/mg'
_label = 'MG'
file_name = 'mg7000'

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
# np.save('data/src/laser/laser.npy', ts)
# %%
# %%
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_acf(ts, lags=84*2, ax =ax1)
sm.graphics.tsa.plot_pacf(ts, lags=84*2, ax=ax2)

# %%
