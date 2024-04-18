import os, sys
# print(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))

from re import S
from numpy.lib.function_base import select
from ray import tune

# from task.ModelSetting import esn_base, cnn_base,esm_base, stat_base
from task.TaskLoader import TaskDataset, Opt
import numpy as np
from task.parser import get_parser
from task.TaskWrapperV1 import Task
# from models.stochastic.cnn import ESM_CNN
import pandas as pd

from models.statistical._setting import autoArima, es , naiveA , naiveL
from models.stochastic.cnn._setting import esm_base, eto_base,convNESN_based
from models.stochastic.esn._setting import esn_base, gesn_base, desn_base, bpsoesn_base, dresn_base,psogesn_base, aeesn_base
from models.stochastic.mlp._setting import scn_base,ielm_base

class Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
    
    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.lag_order = 180
        self.info.period = 60
        self.info.batch_size = 1024
        
    def sub_config(self):
        ts_tuple_list = []
        self.info.series_name = ['SFTS']
        
        for name in self.info.series_name:
            raw_ts = np.load(
                'data/synthetic/laser/laser.npy').reshape(-1,)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)
        

class esm(esm_base):
    def task_modify(self):
        # self.innerTuning = False
        # self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
        # self.tuning.nonlinearity = tune.choice(['sigmoid', 'tanh', 'relu'])
        self.innerTuning = True
        self.hyper.channel_size = 40
        # self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
        # self.tuning.nonlinearity = tune.choice(['sigmoid', 'tanh', 'relu'])

class esn(esn_base):        
    def task_modify(self):
        # self.tuning.hidden_size = tune.qrandint(50, 1000, 25)
        self.hyper.hidden_size = 180 * 3
        
        
class gesn(gesn_base):
    def task_modify(self):
        # self.tuning.hidden_size = tune.qrandint(5, 80, 5)
        # self.tuning.branch_size =  tune.qrandint(4, 30, 2)
        self.hyper.hidden_size = 5
        self.hyper.branch_size = 40

class desn(desn_base):
    def task_modify(self):
        self.tuning.hidden_size = tune.qrandint(5, 80, 5)
        self.tuning.num_layers = tune.qrandint(2, 16, 2)

class arima(autoArima):        
    def task_modify(self):
        self.hyper.max_length = 180 
        
class holt(es):
    def task_modify(self):
        self.hyper.max_length = 180 


class eto(eto_base):
    def task_modify(self):    
        self.hyper.max_cells = 50
        self.hyper.patience_bos= 40 
        self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 50, 5)
        self.hyper.cnn.tuning.kernel_size = tune.qrandint(30, 90, 10)
        self.hyper.cTrain_info.max_epoch = 500
        self.hyper.cTrain_info.metric_loss = 'training'
        
class eto_st(eto_base):
    def task_modify(self):    
        self.hyper.max_cells = 50
        self.hyper.patience_bos= 40 
        self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 50, 5)
        self.hyper.cnn.tuning.kernel_size = tune.qrandint(30, 90, 10)
        self.hyper.cnn.tuner.num_samples = 1
        self.hyper.esn.tuner.num_samples = 1
        self.hyper.esc.tuner.num_samples = 1
        self.hyper.ces.tuner.num_samples = 1
        self.hyper.cTrain_info.max_epoch = 500
        self.hyper.cTrain_info.metric_loss = 'training'
        self.hyper.rTune = False

class scn(scn_base):
    def task_modify(self):
        self.hyper.hidden_size = 400

class ielm(ielm_base):
    def task_modify(self):
        self.hyper.hidden_size = 400

class bpsoesn(bpsoesn_base):
    def task_modify(self):
        self.tuner.num_samples = 200
        
        task_esn = esn()
        self.hyper.esn_opts.hyper.update(task_esn.hyper)
        self.hyper.esn_opts.tuning.update(task_esn.tuning,  ignore_unk=True)
        self.hyper.esn_opts.tuner.update(self.tuner, ignore_unk=True)
        
        self.hyper.esn_opts.hyper.hidden_size = 500

class dresn(dresn_base):
    def task_modify(self):
        self.hyper.branch_size = 50
        
        self.tuner.num_samples = 10 # set as 1 to enable grid search in randn algo.
        self.tuner.algo = 'rand'
        self.tuning.reg_lambda = tune.choice([1e-9, 1e-7,1e-5,1e-3,1e-1,1e1,1e3,1e5,1e7,1e9])
        self.tuning.Ng1=tune.choice([5,10,15,20,25])
        self.tuning.Ng2=tune.choice([3,5,7,9,11])


class psogesn(psogesn_base):
    def task_modify(self):
        self.hyper.readout_steps = 120
        self.hyper.pso_iters = 2


class cmesn(convNESN_based):
    def task_modify(self):
        self.hyper.training_epochs = 50
        

class aeesn(aeesn_base):        
    def task_modify(self):
        self.tuner.resource = {
            "cpu": 30,
            "gpu": 0.5  # set this for GPUs
        }
        self.tuner.num_samples = 20
        
class fpsogesn(psogesn_base):
    def task_modify(self):
        self.hyper.readout_steps = 120    
        self.hyper.hidden_size = 5
        self.hyper.branch_size = 40
        # self.hyper.pso_cpu = 30
        # self.hyper.pso_gpu = 0.25
        self.hyper.pso_iters = 5


                        
if __name__ == "__main__":

    parser = get_parser(parsing=False)
    args = parser.parse_args()
    
    args.cuda = True
    args.datafolder = 'exp/eto/sim'
    args.exp_name = 'paper.eto.sim'
    args.dataset = 'sfts'
    args.H = 1
    args.rep_times = 15
    
    # args.test = True
    # args.model = 'fpsogesn'
    # args.rep_times = 1
    
    task = Task(args)
    # task.tuning()
    task.conduct()
    # task.outlier_check()

    args.metrics = ['rmse', 'smape', 'nrmse']
    task.evaluation(args.metrics, remove_outlier=False)