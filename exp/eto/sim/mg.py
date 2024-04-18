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
from models.stochastic.cnn._setting import esm_base, eto_base, convNESN_based
from models.stochastic.esn._setting import esn_base, gesn_base, desn_base, bpsoesn_base, dresn_base,psogesn_base, aeesn_base
from models.stochastic.mlp._setting import scn_base, ielm_base

import torch

class Data(TaskDataset):
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
        

class esm(esm_base):
    def task_modify(self):
        # self.innerTuning = False
        # self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
        # self.tuning.nonlinearity = tune.choice(['sigmoid', 'tanh', 'relu'])
        # self.tuning.p_size = tune.qrandint(2, 330, 1)   
        self.innerTuning = True
        self.hyper.channel_size = 20
        
class esn(esn_base):        
    def task_modify(self):
        # self.tuning.hidden_size = tune.qrandint(50, 1000, 25)
        self.hyper.hidden_size = 84 * 3        
        
class gesn(gesn_base):
    def task_modify(self):
        # self.tuning.hidden_size = tune.qrandint(5, 80, 5)
        # self.tuning.branch_size =  tune.qrandint(4, 30, 2)
        self.hyper.hidden_size = 5
        self.hyper.branch_size = 20



class desn(desn_base):
    def task_modify(self):
        self.tuning.hidden_size = tune.qrandint(5, 100, 5)
        self.tuning.num_layers = tune.qrandint(2, 16, 2)

class arima(autoArima):        
    def task_modify(self):
        self.hyper.max_length = 1000 
        
class holt(es):
    def task_modify(self):
        self.hyper.max_length = 1000 


class eto(eto_base):
    def task_modify(self):    
        self.hyper.max_cells = 30
        self.hyper.patience_bos= 20 
        self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 100, 5)
        self.hyper.cnn.tuning.kernel_size = tune.qrandint(50, 500, 50)

class scn(scn_base):
    def task_modify(self):
        self.hyper.hidden_size = 100

class ielm(ielm_base):
    def task_modify(self):
        self.hyper.hidden_size = 100
        
class bpsoesn(bpsoesn_base):
    def task_modify(self):
        self.tuner.num_samples = 200
        
        task_esn = esn()
        self.hyper.esn_opts.hyper.update(task_esn.hyper)
        self.hyper.esn_opts.tuning.update(task_esn.tuning,  ignore_unk=True)
        self.hyper.esn_opts.tuner.update(self.tuner, ignore_unk=True)
        
        self.hyper.esn_opts.hyper.hidden_size = 700
        

class dresn(dresn_base):
    def task_modify(self):
        self.hyper.branch_size =100
        
        self.tuner.preTuning_model_path = 'trial/paper.eto.sim/minmax/mg/fit/series0/h84/dresn/tuner/series0.best.pt'
        
        # pT_hyper refers to the proposed method in the cor. paper: Dynamical regularized echo state network for time series prediction. 
        pT_hyper = Opt()
        pT_hyper.reg_lambda = 1e-4
        pT_hyper.Ng1 = 20
        pT_hyper.Ng2 = 5
        torch.save(pT_hyper, self.tuner.preTuning_model_path)


class psogesn(psogesn_base):
    def task_modify(self):
        self.hyper.readout_steps = 666

class cmesn(convNESN_based):
    def task_modify(self):
        self.hyper.training_epochs = 50
        self.hyper.batch_size = 256

class aeesn(aeesn_base):        
    def task_modify(self):
        self.tuner.resource = {
            "cpu": 30,
            "gpu": 0.5  # set this for GPUs
        }
        self.tuner.num_samples = 20
        
class fpsogesn(psogesn_base):
    def task_modify(self):
        self.hyper.readout_steps = 666    
        self.hyper.hidden_size = 5
        self.hyper.branch_size = 20
        self.hyper.pso_iters = 5
        
if __name__ == "__main__":

    parser = get_parser(parsing=False)
    parser.add_argument('-mode', type=str, default='min' )
    parser.add_argument('-met', type=float, )
    parser.add_argument('-re',default=False, action='store_true', help='experiment with cuda')
    args = parser.parse_args()
    
    args.cuda = True
    args.datafolder = 'exp/eto/sim'
    args.exp_name = 'paper.eto.sim'
    args.dataset = 'mg'
    args.H = 84
    args.rep_times = 15
    
    # args.clean = False
    # args.test = True
    # args.model = 'psogesn'
    # args.rep_times = 1
    
    task = Task(args)
    # if args.test is False:
    #     task.tuning()
    task.conduct()
    # task.outlier_check()

    args.metrics = ['rmse', 'smape', 'nrmse']
    task.evaluation(args.metrics, remove_outlier=False)