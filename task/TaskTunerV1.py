'''
Attention !
************
For stochastic model, e.g. ESN, DESN,..., etc. 
They are trained only once that are solving their output-weight in a close form manner. Thus the schedulers based Tuner cannot be implemented into tuning these models, that the tuner will sample the hypers from the config (tuning.dict) only once, and will not tuning sequentially further, causing the schedulers that is to control the training epoch of the one trail is meaningless. 
'''

import os
import sys
from numpy import not_equal

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# import json

import torch

import ray
from task.TaskLoader import Opt
# from ray.tune.suggest.bohb import TuneBOHB
# from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.schedulers import PopulationBasedTraining 
# https://arxiv.org/pdf/1711.09846.pdf.
from ray.tune.schedulers.pb2 import PB2 
# pip install GPy sklearn
# https://arxiv.org/abs/2002.02518 (NIPS 2020)
from ray import tune
from ray.air import session, FailureConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
# from ray.air.checkpoint import Checkpoint

from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.basic_variant import BasicVariantGenerator

import nevergrad as ng

import importlib
# import torch

# import pandas as pd

# import logging


class taskTuner(Opt):
    def __init__(self, opts = None):
        super().__init__()

        if opts is not None:
            self.merge(opts)

        self.points_to_evaluate = []
        if 'points_to_evaluate' in self.tuner.dict:
            self.points_to_evaluate = self.tuner.points_to_evaluate
            assert len(self.points_to_evaluate) > 0

        self.best_config = Opt()

    def search_ax(self,):
        self.tuner.name = 'Bayes_Search'
        # self.tuner.algo = 'algo'
    
        ax_search = ConcurrencyLimiter(AxSearch(metric=self.metric, mode='min',verbose_logging = False), max_concurrent=6)

        return ax_search
    
    def search_pso(self,):
        """
        https://github.com/facebookresearch/nevergrad.
        """
        self.tuner.name = 'PSO_Search'
        # self.tuner.algo = 'pso'
        
        _popsize= min((20, self.tuner.num_samples // 10))
        
        ng_search = NevergradSearch(
            optimizer=ng.optimizers.ConfiguredPSO(
                popsize= _popsize
                ),
            metric=self.metric,
            mode="min",
            points_to_evaluate=self.points_to_evaluate
            )
        return ng_search
    

    def search_tpe(self,):
        '''Tree-structured Parzen Estimator
        https://docs.ray.io/en/master/tune/examples/optuna_example.html
        '''
        self.tuner.name = 'TPE_Search'
        # self.tuner.algo = 'tpe'
    
        tpe_search = ConcurrencyLimiter(
            OptunaSearch(
                metric=self.metric, mode='min',points_to_evaluate=self.points_to_evaluate
                ), 
            max_concurrent=6
            )
        # mute the warning and info in the belowing loggers.
        # for logger_name in ['ax.core.parameter', 'ax.core.parameter','ax.service.utils.instantiation','ax.modelbridge.dispatch_utils']:
        #     logging.getLogger(logger_name).setLevel(logging.ERROR)
        return tpe_search
    
    def search_randn(self,):
        self.tuner.name = 'Rand_Search'
        # self.tuner.algo = 'rand'
    
        rad_search = BasicVariantGenerator(max_concurrent=4)
        # mute the warning and info in the belowing loggers.
        # for logger_name in ['ax.core.parameter', 'ax.core.parameter','ax.service.utils.instantiation','ax.modelbridge.dispatch_utils']:
        #     logging.getLogger(logger_name).setLevel(logging.ERROR)
        return rad_search
    
    # def search_bohb(self,):
    #     """
    #     More efficiency with early stop via Bayesian Optimization HyperBand .
    #     """
    #     self.config_trans()
    #     algo = TuneBOHB(max_concurrent=4)
    #     sched = HyperBandForBOHB()
    #     analysis = tune.run(
    #         self.fitness,
    #         metric="best_loss",
    #         mode="min",
    #         name='BOHB_Search',
    #         search_alg=algo,
    #         scheduler=sched,
    #         config=self.tuning,
    #         resources_per_trial=self.resource,
    #         num_samples=self.tuner.num_samples,
    #         # local_dir=self.opts.task_dir,
    #         verbose=1
    #     )
    #     return analysis


class StocHyperTuner(taskTuner):
    '''
    https://docs.ray.io/en/master/tune/examples/tune-pytorch-cifar.html
    '''
    def __init__(self, opts, logger, subPack):
        # self.xxxx = opts.xxxx
        # super().__init__(self.opts.tunings,search_times)
        super().__init__(opts)

        self.logger = logger
        
        # self.hyper.H = subPack.H
        # self.hyper.sid = subPack.index
        # self.hyper.sid_name = subPack.name
        self.hyper.cid = 'T'
        
        self.train_data = subPack.train_data
        self.valid_data = subPack.valid_data
        self.batch_size = subPack.batch_size
        
        # for testing
        # self.tuner.metric = 'vrmse' # to do: comment
        self.metric = 'vrmse'
        
        if 'algo' not in self.tuner.dict:
            self.algo_name = 'rand'
        elif self.tuner.algo not in ['tpe','pso', 'rand', 'grid']:
            raise ValueError('Non supported tuning algo: {}'.format(self.tuner.algo))
        else:
            self.algo_name = self.tuner.algo
        
        if 'num_samples' not in self.tuner.dict:
            self.tuner.num_samples = 20
            if self.algo_name == 'grid':
                self.tuner.num_samples = 1
        
        if 'resource' not in self.tuner.dict:
            self.resource = {
            "cpu": 10, 
            "gpu": 1  # set this for GPUs
        }
        else:
            self.resource = self.tuner.resource
            
        self.loss_upper_bound = 99999
        
        if self.algo_name == 'tpe':
            self.algo_func = self.search_tpe()
        elif self.algo_name == 'pso':
            self.algo_func = self.search_pso()
        elif self.algo_name == 'rand' or self.algo_name == 'grid':
            self.algo_func = self.search_randn()

    def once_sample(self,):
        config = Opt(init=self.tuning)
        for key in self.tuning.dict:
            config.dict[key] = self.tuning.dict[key].sample()
        
        _hyper = Opt()
        _hyper.merge(self.hyper)
        _hyper.update(config) # Using ignore_unk will be very risky
        model = importlib.import_module(self.import_path)
        model = getattr(model, self.class_name)
        model = model(_hyper, self.logger)
        fit_info = model.xfit(self.train_data, self.valid_data,)
        trmse, vrmse = fit_info.trmse, fit_info.vrmse
        trmse = trmse if trmse < self.loss_upper_bound else self.loss_upper_bound
        vrmse = vrmse if vrmse < self.loss_upper_bound else self.loss_upper_bound        
        
        metric_dict = {
            'trmse': trmse,
            'vrmse': vrmse,
        }
        self.best_result = metric_dict[self.metric]
        self.best_config.merge(config)
        # self.logger.info("Best config is:", self.best_config.dict)
        return self.best_config

    def _conduct(self,):
        
        func_data = Opt()
        func_data.logger = self.logger
        func_data.merge(self,['hyper', 'import_path', 'class_name','train_data', 'valid_data'])
        
        
        ray.init(num_cpus=30)
        # self.tuner.num_samples = 80
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(TuningSNNCell, data=func_data), 
                resources=self.resource),
            param_space=self.tuning.dict,
            tune_config=
            tune.TuneConfig(
            # name=self.algo_name,
            search_alg=self.algo_func,
            # resources_per_trial=self.resource,
            metric=self.metric,
            mode="min",
            num_samples=self.tuner.num_samples,
            # storage_path=self.tuner.dir,
            # verbose=1,
            # raise_on_failed_trial = False
            ),
            run_config=RunConfig(
                name=self.algo_name,
                storage_path=self.tuner.dir,
                verbose=1,
                failure_config=FailureConfig(max_failures=self.tuner.num_samples // 2),
                stop={'training_iteration':1}, # Beacasue of the stochastic mechanism, the weight training iteration of the stochastic model is only one. 
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=0,
                    checkpoint_at_end = False
                ),
                sync_config=tune.SyncConfig(
                    syncer=None
                )                
            )
        )
        
        results = tuner.fit() 
            
        df = results.get_dataframe()
        df.to_csv(os.path.join(self.tuner.dir, '{}.trial.csv'.format(self.algo_name)))
        ray.shutdown()
        
        best_result = results.get_best_result(self.metric, 'min')
        self.best_config.merge(best_result.config)
        self.best_result = best_result.metrics
        # self.logger.info("Best config is:", self.best_config.dict)
        
        return self.best_config

    def conduct(self,):
        if self.tuner.num_samples == 1 and self.algo_name == 'rand':
            self.best_config = self.once_sample()
        else:
            self.best_config = self._conduct()
        
        return self.best_config
    

class TuningSNNCell(tune.Trainable):
    '''
    Trainable class of tuning stochastic neural network
    '''
    def setup(self, config, data=None):
        self.train_data = data.train_data
        self.valid_data = data.valid_data
        self.base_hyper = data.hyper
        self.import_path = data.import_path
        self.class_name = data.class_name
        self.logger = data.logger
        
        _hyper = Opt()
        _hyper.merge(self.base_hyper)
        _hyper.update(config) # Using ignore_unk will be very risky
        
        self.sample_hyper = _hyper
        _model = importlib.import_module(self.import_path)
        _model = getattr(_model, self.class_name)
        self.sample_model = _model(self.sample_hyper, self.logger)
    
    def step(self,):
        fit_info = self.sample_model.xfit(self.train_data, self.valid_data,)
        trmse, vrmse = fit_info.trmse, fit_info.vrmse
        
        return {
            'trmse': trmse,
            'vrmse': vrmse,
        }
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.sample_model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.sample_model.load_state_dict(torch.load(checkpoint_path))    