import os, sys
from task.TaskLoader import TaskDataset, Opt
import numpy as np
import pandas as pd

class GEF2017_Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        # self.info.num_series = 1
        self.info.lag_order = 24*7
        self.info.period = 24
        self.info.batch_size = 256
        
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['ME','NH','VT','CT','RI','SEMA','WCMA','NEMA']
        
        for name in self.info.series_name:
            df = pd.read_excel('data/real/gef/2017_smd_hourly.xlsx',sheet_name=name, index_col=None,header=0)
            raw_ts = df['RT_Demand'].values

            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)
        
class GEF2017ISO_Data(GEF2017_Data):
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['ISO NE CA']
        for i, name in enumerate(self.info.series_name):
            df = pd.read_excel('data/real/gef/2017_smd_hourly.xlsx',sheet_name=name, index_col=None,header=0)
            data = df['RT_Demand']
            if data.isnull().any():
                data= data.interpolate()
            raw_ts = data.values.reshape(-1, ) / 1000

            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)

class NSW2019_Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        # self.info.num_series = 1
        self.info.lag_order = 24*7
        self.info.period = 24
        self.info.batch_size = 256
        
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['NSW2019']
        
        for name in self.info.series_name:
            raw_ts = np.load('data/real/nsw2017_19/nsw_data/loadDemand.2019.npy').reshape(-1,)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)

class PM_Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        # self.info.num_series = 1
        self.info.lag_order = 24*7
        self.info.period = 24
        self.info.batch_size = 1024
        
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['Guangzhou','Chengdu','Shanghai']
        
        for name in self.info.series_name:
            raw_yid = []
            for yid in range(2010,2016):
                df = pd.read_csv(os.path.join('data/real/pm2.5', '{}'.format(yid),'{}.post.csv'.format(name)), header=0, index_col=[0])
                data = df['PM_US Post']
                null_num = len(data[data.isnull()].index)
                if null_num > 0:
                    data = data.interpolate()
                raw_ts_y = data.values.reshape(-1, )
                
                raw_yid.append(raw_ts_y)
            
            raw_ts = np.concatenate(raw_yid)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)

class Sunspots_Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.lag_order = 12*3
        self.info.period = 6
        self.info.batch_size = 512
        
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['Sunspots']
        
        for name in self.info.series_name:
            df = pd.read_csv('data/real/Sunspots/Sunspots.txt',header=None, index_col=None)
            raw_ts = df[0].values.reshape(-1,)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)