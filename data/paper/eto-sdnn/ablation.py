# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# from numpy.lib.function_base import select
from ray import tune
from task.TaskLoader import Opt, TaskDataset
from data.base import esn_base, cnn_base, nn_base, ice_base, stat_base
import numpy as np


class ablation_data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.cov_dim = 0
        self.info.steps = 180
        self.info.input_dim = 1
        self.info.period = 60
        self.info.batch_size = 4096

    def sub_config(self,):
        self.seriesPack = []

        for i in range(self.info.num_series):
            raw_ts = np.load(
                'data/src/laser/laser.npy').reshape(-1,)
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = 'laser'
            sub.H = self.info.H
            sub.merge(self.info)
            
            self.seriesPack.append(sub)


        # self.hyper.cnn.tuning.fc_io = tune.choice([True]) # default true

# class iced(ice):
#     def base_modify(self):
#         self.import_path = 'models/stochastic/cnn/ice/deeper/arch_d.py'
#         self.class_name = 'ICESN'
        
# For the ablation study, the ice has been divided into three modules, i.e., pre-tuning, micro-training, readout-tuning, which are abbreviated as p, m, and r respectively.

class wider(ice_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/ice/wider/arch.py'
        self.class_name = 'ICESN'

    def hyper_modify(self):
        self.hyper.patience_bos = 40
        self.hyper.max_cells = 50
                        
        self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 50, 5)
        # self.hyper.esn.tuning.fc_io = tune.choice([True])   # default true
                     
        self.hyper.cnn.hyper.kernel_size = 60
        self.hyper.cnn.tuning.kernel_size = tune.qrandint(30, 90, 10)
        
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

'''
The ablation study on laser dataset
'''
        
class pt(wider):
    def ablation_modify(self):
        self.hyper.cnn.tuner.iters = 16 
        self.hyper.esn.tuner.iters = 16
        self.hyper.esc.tuner.iters = 16
        self.hyper.ces.tuner.iters = 16


class st(wider):
    def __init__(self):
        wider.__init__(self)
        self.ablation_addition()
    
    def ablation_addition(self):
        self.hyper.cTrain = True
        self.hyper.cTrain_info = Opt()
        self.hyper.cTrain_info.training_name = 'naive'
        self.hyper.cTrain_info.max_epoch = 50

class rt(wider):
    def __init__(self):
        wider.__init__(self)
        self.ablation_addition()
    
    def ablation_addition(self):
        self.hyper.rTune = True
        self.hyper.ho = Opt()
        self.hyper.ho.tuning = Opt()
        self.hyper.ho.tuning.lambda_reg = tune.uniform(0, 5)
        self.hyper.ho.tuner = Opt()
        self.hyper.ho.tuner.iters = 20

class pr(wider):
    def ablation_modify(self):
        self.hyper.cnn.tuner.iters = 16 
        self.hyper.esn.tuner.iters = 16
        self.hyper.esc.tuner.iters = 16
        self.hyper.ces.tuner.iters = 16
        
        self.hyper.rTune = True
        self.hyper.ho = Opt()
        self.hyper.ho.tuning = Opt()
        self.hyper.ho.tuning.lambda_reg = tune.uniform(0, 5)
        self.hyper.ho.tuner = Opt()
        self.hyper.ho.tuner.iters = 20

class ps(wider):
    def ablation_modify(self):
        self.hyper.cnn.tuner.iters = 16 
        self.hyper.esn.tuner.iters = 16
        self.hyper.esc.tuner.iters = 16
        self.hyper.ces.tuner.iters = 16
        
        self.hyper.cTrain = True
        self.hyper.cTrain_info = Opt()
        self.hyper.cTrain_info.training_name = 'naive'
        self.hyper.cTrain_info.max_epoch = 50

class sr(st):
    def ablation_addition(self):
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