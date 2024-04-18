#%%
# https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data
import pandas as pd
import numpy as np
import os 
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import matplotlib
# %%
data_folder = 'data/real/nsw2017_19'
year_fold = 'nsw_raw_data'

data_path = Path(os.path.join(data_folder,year_fold))
data = [x for x in data_path.iterdir() if x.is_file() ]
# %%
price_folder = os.path.join(data_folder, 'nsw_data')
if os.path.exists(price_folder):
    shutil.rmtree(price_folder)

os.makedirs(price_folder)

# %%
def get_data(data, year, m_tag):
    y_mon = [x for x in data if year in str(x) and x.name.split('_')[3][-2:] == m_tag][0]
    year = y_mon.name.split('_')[3][:-2]
    mon = y_mon.name.split('_')[3][-2:]
    df = pd.read_csv(str(y_mon), index_col='SETTLEMENTDATE', parse_dates=True)
    mdemand = df['TOTALDEMAND']
    mdemand = mdemand.values
    mprice = df['RRP'].values

    return df, mdemand, mprice
# %%
for year in [2017, 2018, 2019]:
    year = str(year)
    
    year_data = []
    for month in range(1,13):
        m_tag = '0{}'.format(month) if month < 10 else str(month)
        
        df, mdemand, mprice = get_data(data, year, m_tag)
        print('-'*50)
        print('Half-hourly:\t {}\t{}\tMax: {:.3f}\tMin: {:.3f} \tMean: {:.3f}'.format(year, m_tag, mprice.max(), mprice.min(), mprice.mean()))
        
        ts = pd.Series(mprice, index = df.index.copy())
        # ts_q1 = ts.quantile(0.25, interpolation='nearest')
        # ts_q3 = ts.quantile(0.75, interpolation='nearest')
        # iqr = ts_q3 - ts_q1
        # r = 5
        
        # # eval_l_bound = ts_q1 - r * iqr
        # eval_u_bound = ts_q3 + r * iqr
        # eval_u_bound = 400
        
        # ts_score= ts.copy()
        # ts_score= ts_score> eval_u_bound 
        
        # print(ts.loc[ts_score == True])
        # ts[ts_score == True] = None
        # ts = ts.interpolate()
        # print(ts.loc[ts_score == True])
        _hprice = ts.resample('1H', offset='0.5H').sum()
        hprice = _hprice.values
        print('Hourly:\t {} \t{} \tMax: {:.3f}\tMin: {:.3f} \tMean: {:.3f}'.format(year, m_tag, hprice.max(), hprice.min(), hprice.mean()))
        
        hprice = hprice.tolist()
        year_data.extend(hprice)

    year_data = np.array(year_data)
    print(year_data.shape)
    # year_data = year_data
    font = {'family': 'Times New Roman',
            'size': '44'}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(1,1, figsize=(32,8), dpi=300)

    ax.set_xlabel('Step', fontsize=44)
    ax.set_ylabel('Value', fontsize=44)
    ax.plot(year_data, label = 'NSW')

    plt.legend(loc='upper left')
    plt.xticks(fontsize = 40)
    plt.yticks(fontsize = 40)

    plt.show()
    fig.savefig(os.path.join(price_folder, 'loadPrice.{}.png'.format(year)), dpi= 300, bbox_inches='tight')
    np.save(os.path.join(price_folder, 'loadPrice.{}'.format(year)), year_data)
        
# %%
for year in [2017, 2018, 2019]:
    year = str(year)
    
    year_data = []
    for month in range(1,13):
        m_tag = '0{}'.format(month) if month < 10 else str(month)
        
        df, mdemand, mprice = get_data(data, year, m_tag)
        print('-'*50)
        print('Half-hourly:\t {}\t{}\tMax: {:.3f}\tMin: {:.3f} \tMean: {:.3f}'.format(year, m_tag, mprice.max(), mprice.min(), mprice.mean()))
        
        ts = pd.Series(mdemand / 1000, index = df.index.copy())
        # ts_q1 = ts.quantile(0.25, interpolation='nearest')
        # ts_q3 = ts.quantile(0.75, interpolation='nearest')
        # iqr = ts_q3 - ts_q1
        # r = 5
        
        # # eval_l_bound = ts_q1 - r * iqr
        # eval_u_bound = ts_q3 + r * iqr
        # eval_u_bound = 400
        
        # ts_score= ts.copy()
        # ts_score= ts_score> eval_u_bound 
        
        # print(ts.loc[ts_score == True])
        # ts[ts_score == True] = None
        # ts = ts.interpolate()
        # print(ts.loc[ts_score == True])
        _hprice = ts.resample('1H', offset='0.5H').mean()
        hprice = _hprice.values
        print('Hourly:\t {} \t{} \tMax: {:.3f}\tMin: {:.3f} \tMean: {:.3f}'.format(year, m_tag, hprice.max(), hprice.min(), hprice.mean()))
        
        hprice = hprice.tolist()
        year_data.extend(hprice)

    year_data = np.array(year_data)
    print(year_data.shape)
    # year_data = year_data
    font = {'family': 'Times New Roman',
            'size': '44'}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(1,1, figsize=(32,8), dpi=300)

    ax.set_xlabel('Step', fontsize=44)
    ax.set_ylabel('Value', fontsize=44)
    ax.plot(year_data, label = 'NSW')

    plt.legend(loc='upper left')
    plt.xticks(fontsize = 40)
    plt.yticks(fontsize = 40)

    plt.show()
    fig.savefig(os.path.join(price_folder, 'loadDemand.{}.png'.format(year)), dpi= 300, bbox_inches='tight')
    np.save(os.path.join(price_folder, 'loadDemand.{}'.format(year)), year_data)
# %%
# import statsmodels.api as sm


ts = np.load('data/real/nsw2017_19/nsw_data/loadPrice.2019.npy')
print(ts.shape)
folder = 'data/real/nsw2017_19/nsw_data'
_label = 'NSW'
file_name = 'nsw'

raw_ts = ts
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
# %%
