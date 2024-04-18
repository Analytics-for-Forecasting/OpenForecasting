# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

# %%
file_name = 'Brent-d'
_label = 'BRENT-daily'
folder = 'data/real/cop'
file_path =  os.path.join(folder,'{}.xls'.format(file_name))
data = pd.read_excel(file_path, sheet_name='Data 1', header= 2, index_col=0, )

data[(data < 0).any(1)] = np.nan
data =  data.interpolate()
ts = data.to_numpy()
ts = ts.reshape(-1,)

print(ts.shape)

raw_ts = ts
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
fig.savefig(os.path.join(folder, '{}.png'.format(file_name)), dpi= 300, bbox_inches='tight')

plt.close()

# %%
data
# %%
data[(data < 0).any(1)] = np.nan
# %%
data[(data < 0).any(1)]

# %%
data[data.isnull().any(1)]
# %%
data =  data.interpolate()
# %%
data[data.isnull().any(1)]
# %%
data
# %%
ts = data.to_numpy()
ts.shape

# npy_file = os.path.join(folder, file_name)
# np.save(npy_file, ts)

# %%
ts = ts.reshape(-1,)
# %%
ts
# %%
ts.shape
# %%
