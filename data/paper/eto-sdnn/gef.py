
from ray import tune
from data.base import esn_base,cnn_base, ice_base, stat_base

from task.TaskLoader import TaskDataset, Opt

import numpy as np
import pandas as pd

class gef_data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
        
    
    def info_config(self):
        self.info.normal = False
        self.info.data_path = 'data/src/gef/2017_smd_hourly.xlsx'
        self.info.series_name = ['ME','NH','VT','CT','RI','SEMA','WCMA','NEMA']
        self.info.num_series = len(self.info.series_name) 
        self.info.steps = 24*7
        self.info.cov_dim = 0
        self.info.period = 24
        self.info.batch_size = 1024

    def sub_config(self,):
        self.seriesPack = []
        
        for i in range(self.info.num_series):
            _name = self.info.series_name[i]
            df = pd.read_excel(self.info.data_path,sheet_name=_name, index_col=None,header=0)
            raw_ts = df['RT_Demand'].values
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = _name
            sub.H = self.info.H
            sub.merge(self.info)            

            self.seriesPack.append(sub)
    

class esn(esn_base):  
    def __init__(self):
        super().__init__()
        
    def base_modify(self,):
        self.import_path='models/stochastic/esn/ESN.py'
        self.class_name = 'EchoStateNetwork'

    def hyper_modify(self):
        self.hyper.leaky_r = 1 # prohibit leaky rule,  if set as 1
        self.hyper.iw_bound = (-0.1,0.1)
        self.hyper.hw_bound = (0.66, 0.99)
        self.hyper.weight_scaling = 0.9
        self.hyper.hidden_size = 480
        self.hyper.nonlinearity = 'sigmoid'
        self.hyper.fc_io = 'on'
        
        
    # def tuning_modify(self):
    #     self.tuning.leaky_r = tune.uniform(0.49, 1)
    #     self.tuning.weight_scaling = tune.uniform(0.6, 0.99)
    #     self.tuning.hidden_size = tune.qrandint(50, 1000, 25)
    #     self.tuning.lambda_reg = tune.uniform(0, 1.50)
    #     self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])

        
class gesn(esn_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/GrowingESN.py'
        self.class_name = 'Growing_ESN'

    def hyper_modify(self):
        self.hyper.leaky_r = 1
        self.hyper.hidden_size = 48
        self.hyper.branch_size = 10
        self.hyper.weight_scaling = 0.9
        self.hyper.hw_bound = (0.66, 0.99)
        self.hyper.nonlinearity = 'sigmoid'

    # def tuning_modify(self):
    #     self.tuning.hidden_size = tune.qrandint(5, 100, 5)
    #     self.tuning.branch_size = tune.qrandint(20, 400, 10)
    #     self.tuning.weight_scaling = tune.uniform(0.6, 0.99)
    #     self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])

class pgesn(gesn):
    def __init__(self):
        super().__init__()
        self.tuner.iters = 12

    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/pgesn.py'
        self.class_name = 'PSO_Gesn'
    
    def ablation_modify(self):
        self.hyper.pso = Opt()
        self.hyper.pso.pop = 4
        self.hyper.pso.max_iter= 4
        self.hyper.pso.w = 0.2
        self.hyper.pso.cp = 0.01
        self.hyper.pso.cg = 0.2

class iesn(gesn):
    def __init__(self):
        super().__init__()
        
    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/GrowingESN.py'
        self.class_name = 'Incremental_ESN'

class desn(esn_base):
    def __init__(self):
        super().__init__()

    def base_modify(self):
        self.import_path = 'models/stochastic/esn/DeepESN.py'
        self.class_name = 'Deep_ESN'

    def hyper_modify(self):
        self.hyper.leaky_r = 0.55
        self.hyper.weight_scaling = 0.9
        self.hyper.iw_bound = (-0.1,0.1)
        self.hyper.hw_bound = (0.66,0.99)
        self.hyper.nonlinearity = 'tanh'
        self.hyper.hidden_size = 100
        self.hyper.num_layers = 10
        
    def tuning_modify(self):
        self.tuning.num_layers = tune.qrandint(2, 16, 2)
        self.tuning.hidden_size = tune.qrandint(5, 150, 5)
        self.tuning.weight_scaling = tune.uniform(0.6, 0.99)   
        self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])





class esm(cnn_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/ESM_CNN.py'
        self.class_name = 'ESM_CNN'

    def hyper_init(self):
        self.hyper.channel_size = 10
        self.hyper.candidate_size = 1
        self.hyper.hw_lambda = 0.5
        self.hyper.p_size = 3
        self.hyper.search = 'greedy'
        self.hyper.tolerance = 0
        self.hyper.nonlinearity = 'sigmoid'
        
    def tuning_modify(self):
        self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
        self.tuning.p_size = tune.qrandint(2, 48, 2)
        self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])

# class es(esm):
#     def __init__(self):
#         super().__init__()
        
#     def base_modify(self):
#         self.import_path = 'models/stochastic/cnn/ESM_CNN.py'
#         self.class_name = 'ES_CNN'
    
#     def tuning_modify(self):
#         self.tuning.kernel_size = tune.qrandint(12, 84, 12)
#         self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
#         self.tuning.p_size = tune.qrandint(2, 48, 2)
#         self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])

        
        
class wider(ice_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/ice/wider/arch.py'
        self.class_name = 'ICESN'
        
    def hyper_modify(self):
        self.hyper.patience_bos = 5
        self.hyper.patience_bos = 10
        self.hyper.max_cells = 20
                        
        self.hyper.esn.tuning.hidden_size = tune.qrandint(48, 168, 12)
        # self.hyper.esn.tuning.fc_io = tune.choice([True])   # default true           
        self.hyper.cnn.hyper.kernel_size = 42
        self.hyper.cnn.tuning.kernel_size = tune.qrandint(12, 84, 12) 
        self.hyper.cnn.tuning.pooling_size = tune.qrandint(2, 48, 2)

    def ablation_modify(self):
        self.hyper.cnn.tuner.iters = 1 # equal to randomly select ones without preTuning
        self.hyper.esn.tuner.iters = 1   
        self.hyper.esc.tuner.iters = 1
        self.hyper.ces.tuner.iters = 1


class wider_pcr(wider):
    def ablation_modify(self):
        self.hyper.cnn.tuner.iters = 16 
        self.hyper.esn.tuner.iters = 16
        self.hyper.esc.tuner.iters = 16
        self.hyper.ces.tuner.iters = 16
        
        self.hyper.cTrain = True
        self.hyper.cTrain_info = Opt()
        self.hyper.cTrain_info.training_name = 'naive'
        self.hyper.cTrain_info.max_epoch = 50
        
        self.hyper.rTune = True
        self.hyper.ho = Opt()
        self.hyper.ho.tuning = Opt()
        self.hyper.ho.tuning.lambda_reg = tune.uniform(0, 5)
        self.hyper.ho.tuner = Opt()
        self.hyper.ho.tuner.iters = 20

class small(wider_pcr):
    def __init__(self):
        super().__init__()
        self.ablation_addition()
        
    def ablation_addition(self):
        self.hyper.arch_choice = ['cnn', 'esn']
        self.hyper.cTrain_info.max_epoch = 100

class naiveL(stat_base):        
    def base_modify(self):
        self.import_path = 'models/statistical/naive.py'
        self.class_name = 'Naive'
    
    def hyper_modify(self):
        self.hyper.method = 'last'
        
class naiveA(naiveL):
    def hyper_modify(self):
        self.hyper.method = 'avg'
        
class naiveP(naiveL):
    def hyper_modify(self):
        self.hyper.method = 'period'  
        
class arima(stat_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/statistical/arima.py'
        self.class_name = 'ARIMA'
    
    def hyper_modify(self):
        self.hyper.period = 168
        self.hyper.refit = True
        
class holy(arima):
    def base_modify(self):
        self.import_path = 'models/statistical/holy.py'
        self.class_name = 'Holy'