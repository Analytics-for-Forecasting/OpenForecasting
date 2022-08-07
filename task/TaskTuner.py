'''
Attention !
************
For stochastic model, e.g. ESN, DESN,..., etc. 
They are trained only once that are solving their output-weight in a close form manner. Thus the schedulers based Tuner cannot be implemented into tuning these models, that the tuner will sample the hypers from the config (tuning.dict) only once, and will not tuning sequentially further, causing the parameters 'tuner.iters' is meaningless. 
'''

import os
import sys
from numpy import not_equal

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# import json
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
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest import ConcurrencyLimiter

import nevergrad as ng

import importlib
import torch

import pandas as pd

import logging


class Tuner(Opt):
    def __init__(self, opts = None):
        super().__init__()

        if opts is not None:
            self.merge(opts)

        self.best_config = None
        
    def conduct(self,):
        analysis = self._conduct()
        self.best_config = analysis.get_best_config()
        self.best_result = analysis.best_result
        print("Best config is:", analysis.best_config)
        # print('Best loss is: {:.4f}'.format(
        #     analysis.best_result['best_loss']))

    def _conduct(self,):
        ray.init(num_cpus=self.tuner.cores)
        self.resource = {
            "cpu": 10,
            "gpu": self.tuner.cards  # set this for GPUs
        }
        analysis = self.search_base()
        return analysis
    
    def search_base(self,):
        analysis = tune.run(
            self.fitness,
            config=self.search_configs,
            num_samples=6
        )
        return analysis

    def fitness(self, config):
        '''Loading the hyper-parameters in config, then return a score'''
        best_score = float('inf')

        for i in range(self.tuner.epochPerIter):
            score = 0
            if score <= best_score:
                best_score = score
            tune.report(best_loss=best_score, current_loss=score)


    def search_pbt(self,):
        """
        More effective via Population Based Bandits. \n
        The primary motivation for PB2 is the ability to find promising hyperparamters with only a small population size. \n
        This algorithm is only suitable for training_based models. 
        """
        self.tuner.name = 'PBT_Search'
        # def LU_tranfer(config):
        #     bounds = {}
        #     for key in config.keys():
        #         if isinstance(config[key], Float):
        #             bounds[key] = [config[key].lower, config[key].upper]
        #     return bounds
        # sched = PB2(
        #     time_attr='training_iteration',
        #     perturbation_interval=5,
        #     hyperparam_bounds = LU_tranfer(self.tuning.dict)
        # )
        sched = PopulationBasedTraining(
            time_attr='training_iteration',
            perturbation_interval=5,
            hyperparam_mutations=self.tuning.dict,
            resample_probability=0.5,
        )        
        analysis = tune.run(
            self.fitness,
            name=self.tuner.name,
            scheduler=sched,
            config=self.tuning.dict,
            resources_per_trial=self.resource,
            metric="best_loss",
            mode="min",
            num_samples=self.tuner.iters,
            local_dir=self.tuner.dir,
            verbose=1
        )
        return analysis  
    
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
    #         num_samples=self.tuner.iters,
    #         # local_dir=self.opts.task_dir,
    #         verbose=1
    #     )
    #     return analysis


class StocHyperTuner(Tuner):
    def __init__(self, opts, logger, subPack):
        # self.xxxx = opts.xxxx
        # super().__init__(self.opts.tunings,search_times)
        super().__init__(opts)

        self.logger = logger
        self.train_loader = subPack.train_loader
        self.valid_loader = subPack.valid_loader
        # for testing
        self.tuner.metric = 'best_vrmse' # to do: comment
        self.metric = self.tuner.metric
        
        if 'iters' not in self.tuner.dict:
            self.tuner.iters = 20

    def _fitness(self, model):
        '''
        config is equal to self.tuning
        '''
        train_loader = self.train_loader
        valid_loader = self.valid_loader

        fit_info = model.xfit(train_loader, valid_loader)
        return fit_info

    def fitness(self, config, checkpoint_dir = None):

        _hyper = Opt()
        _hyper.merge(self.hyper)
        _hyper.update(config) 
        model = importlib.import_module(self.import_path)
        model = getattr(model, self.class_name)
        model = model(_hyper, self.logger)

        if checkpoint_dir:
            print("Loading from checkpoint.")
            path = os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["model_state_dict"])

        best_trmse = float('inf')
        best_vrmse = float('inf')
        
        for epoch in range(self.tuner.epochPerIter):
            fit_info = self._fitness(model)
            trmse, vrmse = fit_info.trmse, fit_info.vrmse
            if trmse <= best_trmse:
                best_trmse = trmse
            if vrmse <= best_vrmse:
                best_vrmse = vrmse
            
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (model.state_dict()), path)
                
            tune.report(
                best_trmse=best_trmse,
                cur_trmse=trmse,
                best_vrmse = best_vrmse,
                cur_vrmse= vrmse)

    def search_ng(self,):
        """
        https://github.com/facebookresearch/nevergrad.
        """
        self.tuner.name = 'PSO_Search'
        self.tuner.algo = 'algo'
        
        ng_search = NevergradSearch(
            optimizer=ng.optimizers.ConfiguredPSO(popsize=20),
            metric=self.metric,
            mode="min",)
        return ng_search
    
    def search_ax(self,):
        self.tuner.name = 'Bayes_Search'
        self.tuner.algo = 'algo'
    
        ax_search = ConcurrencyLimiter(AxSearch(metric=self.metric, mode='min',verbose_logging = False), max_concurrent=4)
        # mute the warning and info in the belowing loggers.
        for logger_name in ['ax.core.parameter', 'ax.core.parameter','ax.service.utils.instantiation','ax.modelbridge.dispatch_utils']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        return ax_search
    
    def algo_run(self, algo):
        if self.tuner.algo == 'algo':
            analysis = tune.run(
            self.fitness,
            name=self.tuner.name,
            search_alg=algo,
            config=self.tuning.dict,
            resources_per_trial=self.resource,
            metric=self.metric,
            mode="min",
            num_samples=self.tuner.iters,
            local_dir=self.tuner.dir,
            verbose=1,
            raise_on_failed_trial = False
        )
        elif self.tuner.algo == 'sche':
            analysis = tune.run(
            self.fitness,
            name=self.tuner.name,
            scheduler=algo,
            config=self.tuning.dict,
            resources_per_trial=self.resource,
            metric=self.metric,
            mode="min",
            num_samples=self.tuner.iters,
            local_dir=self.tuner.dir,
            verbose=1,
            raise_on_failed_trial = False
        )
        else:
            assert False
        
        return analysis

    
    def save(self, analysis):
        all_results = analysis.results
        ids = list(all_results.keys())
        
        results = pd.DataFrame()
        for id in ids:
            result = all_results[id]
            df = pd.DataFrame()
            record = {}
            for key in ['trial_id', 'training_iteration','best_trmse','cur_trmse','best_vrmse','cur_vrmse']:
                record[key] = result[key]
            df = pd.DataFrame(record, index=[0])

            _df = pd.DataFrame(result['config'],index=[0])
            df = pd.concat([df, _df],axis= 1)
            results = pd.concat([results, df]).reset_index(drop=True)
        results.to_csv(os.path.join(self.tuner.dir, '{}.trial.csv'.format(self.tuner.name)))
    
    def _conduct(self):
        ray.init(num_cpus=20)
        self.resource = {
            "cpu": 10,
            "gpu": 1  # set this for GPUs
        }
        # self.tuner.iters = 80
        algo = self.search_ax()
        analysis = self.algo_run(algo)
        self.save(analysis)
        ray.shutdown()
        return analysis