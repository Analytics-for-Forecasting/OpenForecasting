import os, sys
from task.TaskLoader import TaskDataset, Opt
import numpy as np
import pandas as pd

class SFTS_Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        # self.info.num_series = 1
        self.info.cov_dim = 0
        self.info.lag_order = 180
        # self.info.input_dim = 1
        self.info.period = 60
        self.info.batch_size = 4096
        
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['laser.D']

        for name in self.info.series_name:            
            raw_ts = np.load(
                'data/synthetic/laser/laser.npy').reshape(-1,)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)
        
class MG17_Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
    
    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.lag_order = 1000
        self.info.period = 17
        self.info.batch_size = 1024
        
    def sub_config(self):
        ts_tuple_list = []
        self.info.series_name = ['MG17']
        
        for name in self.info.series_name:
            raw_ts = np.load(
                'data/synthetic/mg/mg.npy').reshape(-1,)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)

class Lorenz_Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.lag_order = 60
        self.info.period = 20
        self.info.batch_size = 512
        
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['Lorenz']
        
        for name in self.info.series_name:
            df = pd.read_csv('data/synthetic/lorenz/Lorenz.txt',header=None, index_col=None)
            raw_ts = df[0].values.reshape(-1,)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)        