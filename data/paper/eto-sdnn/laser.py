# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# from numpy.lib.function_base import select
from ray import tune
from task.TaskLoader import Opt, TaskDataset
from data.base import esn_base, cnn_base, nn_base, ice_base, stat_base
import numpy as np


class laser_data(TaskDataset):
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


class esn(esn_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/ESN.py'
        self.class_name = 'EchoStateNetwork'
        
    def hyper_modify(self):
        self.hyper.leaky_r = 1
        self.hyper.iw_bound = (-0.1,0.1)
        self.hyper.hw_bound = (0.66, 0.99)
        self.hyper.weight_scaling = 0.9
        self.hyper.hidden_size = 500
        self.hyper.nonlinearity = 'sigmoid'
        self.hyper.fc_io = 'on'

        
class gesn(esn_base):
    def __init__(self):
        super().__init__()

    def base_modify(self):
        self.import_path = 'models/stochastic/esn/GrowingESN.py'
        self.class_name = 'Growing_ESN'

    def hyper_modify(self):
        self.hyper.leaky_r = 1
        self.hyper.hidden_size = 5
        self.hyper.branch_size = 100
        self.hyper.weight_scaling = 0.9
        self.hyper.hw_bound = (0.66, 0.99)
        self.hyper.nonlinearity = 'sigmoid'

    # def tuning_modify(self):
    #     self.tuning.hidden_size = tune.qrandint(5, 80, 5)
    #     self.tuning.branch_size = tune.qrandint(4, 30, 2)
    #     self.tuning.weight_scaling = tune.uniform(0.6, 0.99)
    #     self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])
        
class iesn(gesn):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/GrowingESN.py'
        self.class_name = 'Incremental_ESN'

    def ablation_modify(self):
        self.hyper.branch_size = 100
        # self.tuning.dict.pop('branch_size') 

    # def tuning_modify(self):
    #     self.tuning.leaky_r = tune.uniform(0.49, 1)
    #     self.tuning.weight_scaling = tune.uniform(0.6, 0.99)
    #     self.tuning.hidden_size = tune.qrandint(50, 1000, 25)
    #     self.tuning.lambda_reg = tune.uniform(0, 1.50)
    #     self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])

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
        self.hyper.num_layers = 5
        self.hyper.hidden_size = 100
        self.hyper.nonlinearity = 'tanh'
        
    def tuning_modify(self):
        self.tuning.num_layers = tune.qrandint(2, 16, 2)
        self.tuning.hidden_size = tune.qrandint(5, 80, 5)
        self.tuning.weight_scaling = tune.uniform(0.6, 0.99)   
        self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])


class pgesn(gesn):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/pgesn.py'
        self.class_name = 'PSO_Gesn'

    def ablation_modify(self):
        self.hyper.pso = Opt()
        self.hyper.pso.pop = 10
        self.hyper.pso.max_iter= 10
        self.hyper.pso.w = 0.2
        self.hyper.pso.cp = 0.01
        self.hyper.pso.cg = 0.2        

class esm(cnn_base):
    def __init__(self):
        super().__init__()

    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/ESM_CNN.py'
        self.class_name = 'ESM_CNN'

    def hyper_init(self):
        self.hyper.channel_size = 100
        self.hyper.candidate_size = 30
        self.hyper.hw_lambda = 0.5
        self.hyper.p_size = 3
        self.hyper.search = 'greedy'
        self.hyper.tolerance = 0
        self.hyper.nonlinearity = 'sigmoid'
        
    def tuning_modify(self):
        self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
        self.tuning.p_size = tune.qrandint(2, 60, 1)
        # self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])

class es(esm):
    def __init__(self):
        super().__init__()

    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/ESM_CNN.py'
        self.class_name = 'ES_CNN'
    
    def tuning_modify(self):
        self.tuning.kernel_size = tune.qrandint(30, 90, 10)
        self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
        self.tuning.p_size = tune.qrandint(2, 60, 1)
        self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])

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



class small(wider_pcr):
    def __init__(self):
        super().__init__()
        self.ablation_addition()
        
    def ablation_addition(self):
        self.hyper.arch_choice = ['cnn', 'esn']
        self.hyper.cTrain_info.max_epoch = 100


class mlp(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MLP.py'
        self.class_name = 'MLP'
        
        self.training = True
        self.arch = 'mlp'
    
    def hyper_init(self):
        self.hyper.hidden_size = 400
        self.hyper.epochs = 1000
        self.hyper.learning_rate = 0.01
        self.hyper.step_lr = 20

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
        self.hyper.period = 180
        self.hyper.refit = True        

class holy(arima):
    def base_modify(self):
        self.import_path = 'models/statistical/holy.py'
        self.class_name = 'Holy'        