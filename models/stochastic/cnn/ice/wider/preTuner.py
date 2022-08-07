from time import sleep
from matplotlib.pyplot import cla
from numpy import fabs
from ray.cloudpickle.cloudpickle import cell_set
from ray.tune.trainable import Trainable
from task.TaskLoader import Opt

from models.stochastic.cnn.ice.wider.basic import close_ho, io_check, Arch_dict

import ray
from ray import tune
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest import ConcurrencyLimiter
import logging

import torch
import os
import torch.nn as nn
import numpy as np

import copy
import gc

import sys
from ray.tune import Analysis
current_module = sys.modules[__name__]

import pickle


class PreTuner(Opt):
    def __init__(self, arch_choice):
        super().__init__()
        
        self.arch_choice = arch_choice
        # self.arch_choice = ['esn']
        # self.arch_choice = ['esn']
        
    def tuning(self, aHyper, data_pack, WidthNum, device):
        '''input: hyper-parameter, data_pack, width_num, device\n
        return: new selected arch_cell, updated data_pack'''
    
        self.arch_config_path = os.path.join(data_pack.series_dir, 'tuner', 'WideNet{}'.format(
            WidthNum), 'Arch.config.pkl')
        
        if os.path.exists(self.arch_config_path):
            with open(self.arch_config_path, 'rb') as arch_pkl:
                arch_config = pickle.load(arch_pkl)
                aHyper.dict[arch_config.name].hyper.device = device    
        else:
            arch_config = Opt()
            arch_config.e_norm = float('inf')
            
            for arch_name in self.arch_choice:
                aHyper.dict[arch_name].hyper.device = device            
                
                tuner_dir = os.path.join(data_pack.series_dir, 'tuner', 'WideNet{}'.format(
                WidthNum), 'Arch.{}'.format(arch_name))
                analysis_dir = os.path.join(tuner_dir,'Bayes_Search')
                if os.path.exists(analysis_dir):
                    analysis = Analysis(analysis_dir)
                    best_config = analysis.get_best_config(metric='e_norm', mode='min')
                    
                    df = analysis.dataframe(metric='e_norm', mode='min')
                    arch_loss = df['e_norm'].min()
                else:
                    # atm = getattr(current_module, '{}Tuner'.format(arch_name))
                    arch_tuner = ArchTuner(aHyper.dict[arch_name]) 
                    arch_tuner.conduct(data_pack, WidthNum)
                    arch_loss = arch_tuner.best_result['e_norm']
                    best_config = arch_tuner.best_config
                
                if arch_loss <= arch_config.e_norm:
                    arch_config.best_config = best_config
                    arch_config.name = arch_name
                    arch_config.e_norm = arch_loss

            # saving the arch_cofig results to its path.
            with open(self.arch_config_path, 'wb') as arch_pkl:
                pickle.dump(arch_config, arch_pkl)

        arch_config, best_arch, fc_io = self.config_to_arch(aHyper.dict[arch_config.name], arch_config, data_pack)

        return arch_config, best_arch, fc_io

    def config_to_arch(self, cOpt, arch_config, data_pack):
        hyper = Opt(init=cOpt.hyper)

        hyper.input_dim = data_pack.input_dim
        
        fc_io = arch_config.best_config['fc_io']
        
        arch_config.best_config.pop('fc_io')
        hyper.update(arch_config.best_config)
        
        arch_config.hyper = hyper
        
        arch_func = Arch_dict[arch_config.name]
        best_arch = arch_func(**hyper.dict).to(hyper.device)
        best_arch.freeze()
        return arch_config, best_arch, fc_io


class ArchTuner(Opt):
    def __init__(self, arch_opts):
        super().__init__()

        self.merge(arch_opts)
    
        # self.tuner.metirc = 'e_norm'
        self.metric = 'e_norm'
        if 'iters' not in self.tuner.dict:
            self.tuner.iters = 16
        else:
            self.tuner.iters = self.tuner.iters

        self.cpus = 24
        self.resource = {
            "cpu": 6,
            "gpu": 0.5  # set this for GPUs
        }
                
        # self.tuner.ho_epochs = 500
        self.arch_name = arch_opts.name
        # self.arch_func = Arch_dict[self.arch_name]

        self.loss_fn = nn.MSELoss()

    def space_sampling(self,):
        config = Opt(init=self.tuning)
        for key in self.tuning.dict:
            config.dict[key] = self.tuning.dict[key].sample()
        return config.dict
    
    def funcData_gen(self, data_pack):
        func_data = Opt()
        with torch.no_grad():
            func_data.d_x = [_i.detach().clone() for _i in data_pack.x] # ensure the arch are selected based on input x
            func_data.d_cate = data_pack.catE.detach().clone()
            func_data.d_catx = data_pack.catX.detach().clone()
            func_data.d_catvx = data_pack.catVX.detach().clone()
            func_data.d_catve = data_pack.catVE.detach().clone()
            
        func_data.merge(self, ['hyper', 'arch_name',
                        'device'])
        func_data.merge(data_pack, ['steps', 'input_dim', 'H', 'xm_size'])
        
        return func_data
    
    def once_sample(self, data_pack, WidthNum):
        self.tuner_dir = os.path.join(data_pack.series_dir, 'tuner', 'WideNet{}'.format(
            WidthNum), 'Arch.{}'.format(self.arch_name))
        if not os.path.exists(self.tuner_dir):
            os.makedirs(self.tuner_dir)
        
        self.best_config = self.space_sampling()
            
        func_data = self.funcData_gen(data_pack)
        
        tuning = Cell_trial(self.best_config, func_data)
        # tuning.setup()
        
        result = tuning.step()
        self.best_result = result
        

    def _conduct(self, data_pack, WidthNum):
        self.tuner_dir = os.path.join(data_pack.series_dir, 'tuner', 'WideNet{}'.format(
            WidthNum), 'Arch.{}'.format(self.arch_name))
        if not os.path.exists(self.tuner_dir):
            os.makedirs(self.tuner_dir)
            
        ray.init(num_cpus=self.cpus, _redis_max_memory=10**9)
        # self.tuner.iters = 80
        algo = self.search_ax()

        # cat_e = torch.cat(self.dataPack.e, dim=0)
        # cat_x = torch.cat(self.dataPack.x, dim=0)

        func_data = self.funcData_gen(data_pack)
        # func_data = Opt()
        # with torch.no_grad():
        #     func_data.d_x = [_i.detach().clone() for _i in data_pack.x] # ensure the arch are selected based on input x
        #     func_data.d_cate = data_pack.catE.detach().clone()
        #     func_data.d_catx = data_pack.catX.detach().clone()

        # func_data.merge(self, ['hyper', 'arch_name',
        #                 'device'])
        # func_data.merge(data_pack, ['steps', 'input_dim', 'H', 'xm_size'])

        analysis = tune.run(
            tune.with_parameters(tuningCell, data=func_data),
            name=self.tuner.name,
            search_alg=algo,
            config=self.tuning.dict,
            resources_per_trial=self.resource,
            metric=self.metric,
            mode="min",
            num_samples=self.tuner.iters,
            local_dir=self.tuner_dir,
            verbose=1,
            stop={
                "training_iteration": self.tuner.iters
            },
            raise_on_failed_trial = False
        )

        self.best_config = analysis.get_best_config()
        self.best_result = analysis.best_result
        ray.shutdown()

    def search_ax(self,):
        self.tuner.name = 'Bayes_Search'
        self.tuner.algo = 'algo'
        ax_search = ConcurrencyLimiter(
            AxSearch(
                metric=self.metric, 
                mode='min', 
                verbose_logging=False
                ), 
            max_concurrent=4)
        # mute the warning and info in the belowing loggers.
        for logger_name in ['ax', 'ax.core.parameter', 'ax.core.parameter', 'ax.service.utils.instantiation', 'ax.modelbridge.dispatch_utils']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        return ax_search

    def conduct(self, data_pack, WidthNum):
        self.device = data_pack.device
        
        if self.tuner.iters  == 1:
            self.once_sample(data_pack, WidthNum)
        else:
            self._conduct(data_pack, WidthNum)



# class tuningCell(Cell_trial,tune.Trainable):
#     def __init__(self,  config=None, logger_creator=None):
#         tune.Trainable.__init__(self, config=None, logger_creator=None)

class tuningCell(tune.Trainable):
    # def __init__(self,  config=None, logger_creator=None):
    #     tune.Trainable.__init__(self, config=None, logger_creator=None)
    
    def setup(self, config, data=None):

        self.d_x = data.d_x
        self.d_cate = data.d_cate
        self.d_catx = data.d_catx
        
        self.d_catvx = data.d_catvx 
        self.d_catve = data.d_catve

        self.device = data.device

        self.hyper = data.hyper  # check xm_size if fc_io; check steps.
        self.arch_name = data.arch_name
        self.steps = data.steps

        
        self.H = data.H

        _hyper = Opt()
        _hyper.merge(self.hyper)
        _hyper.fc_io = None
        _hyper.update(config)
        _hyper.input_dim = data.input_dim # must check this for arch mixed-up
        
        self.fc_io = _hyper.fc_io # put the fc_io into the tuning
 
        self.xm_size = data.xm_size

        _hyper.dict.pop('fc_io')
        self.model = Arch_dict[self.arch_name](**_hyper.dict).to(self.device)
        self.model.freeze()
        # self.ho = self.gen_ho(_hyper)
        # self.ho.freeze()

        self.loss_fn = nn.MSELoss()

    # def gen_ho(self, _hyper):
    #     if self.arch_name == 'cnn':
    #         map_size = cnn_map(_hyper, self.steps)
    #     elif self.arch_name == 'esn':
    #         map_size = rnn_map(_hyper)
    #     else:
    #         raise ValueError('Non-supported arch type: {}'.format(self.arch_name)
    #                          )

    #     if self.fc_io:
    #         map_size += self.xm_size
    #     ho = fcLayer(map_size, self.H, self.device)
    #     ho.freeze()
    #     return ho


    def step(self,):
        cat_h = []
        for _x in self.d_x:
            _, h = self.model(_x)
            cat_h.append(h)
        cat_h = torch.cat(cat_h, dim=0)
        cat_h = io_check(self.fc_io, cat_h, self.d_catx)

        # weight, bias = close_ho(self.device, cat_h, self.d_cate)
        # self.ho = fcLayer(cat_h.size(1), self.d_cate.size(1), self.device)
        # self.ho.update(weight, bias)
        # self.ho.freeze()
        try:
            self.ho = close_ho(self.device, cat_h, self.d_cate)
        except:
            self.ho = close_ho(self.device, cat_h, self.d_cate, lambda_reg=10)
        # cat_p = self.ho(cat_h)
        # loss = self.loss_fn(cat_p, self.d_cate).item()
        
        _, cat_vh = self.model(self.d_catvx)
        cat_vh = io_check(self.fc_io, cat_vh, self.d_catvx)
        cat_vp = self.ho(cat_vh)
        
        
        # cat_e = self.d_cate - cat_p
        cat_e = self.d_catve - cat_vp
        e_norm = torch.linalg.matrix_norm(cat_e).item()

        return {
            # "loss": loss,
            "e_norm": e_norm
        }


class Cell_trial(tuningCell):
    def __init__(self, config = None, data = None):
        self.setup(config,data)
        
    
