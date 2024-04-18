# %%
import pandas as pd
import numpy as np
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib

# %%

folder = 'data/real/SML2011'
file_name = 'DATA-1'
_label = 'SML (Data-1)'

df = pd.read_csv(os.path.join('data/real/SML2011/','NEW-{}.T15.txt'.format(file_name)),header=0, sep=' ')
data = df['3:Temperature_Comedor_Sensor']
if data.isnull().any():
    data= data.interpolate()
raw_ts = data.values.reshape(-1, )

font = {'family': 'Times New Roman',
        'size': '22'}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(1,1, figsize=(12,4), dpi=300)

ax.set_xlabel('Step', fontsize=22)
ax.set_ylabel('Value', fontsize=22)
ax.plot(raw_ts, label = _label)

plt.legend(loc='upper left')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()
fig.savefig(os.path.join(folder, 'sml.{}.png'.format(file_name)), dpi= 300, bbox_inches='tight')

plt.close()


# %%

# %%
2764 + 1373

# %%
