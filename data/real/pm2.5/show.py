# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
# %%


file_name = 'shanghai' 
_label = 'PM2.5 (Shanghai)'
folder ='data/real/pm2.5'

raw_yid = []
for yid in range(2010,2016):
    df = pd.read_csv(os.path.join('data/real/pm2.5', '{}'.format(yid),'{}.post.csv'.format(file_name)), header=0, index_col=[0])
    data = df['PM_US Post']
    null_num = len(data[data.isnull()].index)
    if null_num > 0:
        data = data.interpolate()
    raw_ts_y = data.values.reshape(-1, )
    
    raw_yid.append(raw_ts_y)

raw_ts = np.concatenate(raw_yid)

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
def pmshow(city, yid, post = True):
    city_path = os.path.join(folder, '{}'.format(yid), '{}.post.csv'.format(city) if post else '{}.csv'.format(city))
    df = pd.read_csv(city_path,header=0, index_col=[0])
    data = df['PM_US Post']
    null_num = len(data[data.isnull()].index)
    print('Null num: {}'.format(null_num))
    
    if null_num > 0:
        data = data.interpolate()
    
    
    data.plot()
    # plt.title('{} {} post {}'.format(city, yid, 'True' if post else 'False'))
    # plt.show()
# %%
# pmshow('Chengdu' , 2010)
# %%
# cid = 'Chengdu'
# cid = 'Guangzhou'
cid = 'Shanghai'
# cid = 'Shenyang'
for yid in range(2015,2016):
    fig = plt.figure(figsize=(16, 9))
    pmshow(cid, yid, True)
    pmshow(cid, yid, False)
    plt.title('{} {} post {}'.format(cid, yid, 'True' ))
    plt.legend(['post true', 'post false'])
    plt.show()
# %%
