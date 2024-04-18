# %%
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib

# %%
folder = 'data/real/stock'
file_name = 'SP'
# _label = file_name
_label = 'S&P 500'
filepath = os.path.join(folder, 'Processed_{}.csv'.format(file_name))
# filepath = 'data/paper.esm/SP500/SP500.csv'
df = pd.read_csv(filepath,index_col='Date',header=0, parse_dates=['Date']).asfreq('1D')

ts = df['Close']
ts = ts.dropna()

raw_ts = ts.values
font = {'family': 'Times New Roman',
        'size': '22'}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(1,1, figsize=(12,4), dpi=300)

ax.set_xlabel('Step', fontsize=22)
ax.set_ylabel('Value', fontsize=22)
ax.plot(raw_ts, label = _label, linewidth=0.9)

plt.legend(loc='upper left')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()
fig.savefig(os.path.join(folder, '{}.png'.format(file_name)), dpi= 300, bbox_inches='tight')

plt.close()

# %%


# %%
