# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
# %%
data_path = 'data/src/gef/2017_smd_hourly.xlsx'

file_name = 'NEMA'
_label = 'GEF ({})'.format(file_name)
folder = 'data/real/gef'

df = pd.read_excel('data/real/gef/2017_smd_hourly.xlsx', sheet_name=file_name, index_col=None, header=0)
raw_ts = df['RT_Demand'].values

# raw_ts = ts
font = {'family': 'Times New Roman',
        'size': '44'}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(1,1, figsize=(32,8), dpi=300)

ax.set_xlabel('Step', fontsize=44)
ax.set_ylabel('Value', fontsize=44)
ax.plot(raw_ts, label = _label)

plt.legend(loc='upper left')
plt.xticks(fontsize = 40)
plt.yticks(fontsize = 40)

plt.show()
fig.savefig(os.path.join(folder, '{}.png'.format(file_name)), dpi= 300, bbox_inches='tight')

plt.close()

# %%
# df['Date'].astype(str)
# # %%
# df['Hr_End'] - 1 
# # %%
# _time = df['Date'].astype(str) + '-' + (df['Hr_End'] - 1 ).astype(str)

# _time = pd.to_datetime(_time, format='%Y-%m-%d-%H')

# _time
# # %%
# df.index = _time

# df['RT_Demand']

# # %%
# df['RT_Demand'].values.shape
# # %%
