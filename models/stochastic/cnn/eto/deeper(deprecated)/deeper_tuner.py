from time import sleep
from matplotlib.pyplot import cla
from numpy import fabs
from ray.cloudpickle.cloudpickle import cell_set
from ray.tune import analysis
from ray.tune.trainable import Trainable
from task.TaskLoader import Opt
from models.stochastic.Base import fcLayer, esnLayer, cnnLayer

import ray
from ray import tune
from ray.tune.search.ax import AxSearch
from ray.tune.suggest import ConcurrencyLimiter
import logging

import torch
import os
import torch.nn as nn
import numpy as np

import gc

import sys
from ray.tune import Analysis

current_module = sys.modules[__name__]

def close_ho(device, Hidden_States, y, reg_lambda=0, ):
    t_hs = torch.cat((torch.ones(Hidden_States.size(0), 1).to(
        device), Hidden_States), dim=1)
    HTH = t_hs.T @ t_hs
    HTY = t_hs.T @ y
    # ridge regression with pytorch
    I = (reg_lambda * torch.eye(HTH.size(0))).to(device)
    A = HTH + I
    # orig_rank = torch.matrix_rank(HTH).item() # torch.matrix_rank is deprecated in favor of torch.linalg.matrix_rank in v1.9.0
    orig_rank = torch.linalg.matrix_rank(A, hermitian=True).item()
    tag = 'Inverse' if orig_rank == t_hs.size(1) else 'Pseudo-inverse'
    if tag == 'Inverse':
        W = torch.mm(torch.inverse(A), HTY).t()
    else:
        W = torch.mm(torch.linalg.pinv(A.cpu()),
                     HTY.cpu()).t().to(device)

    bias = nn.Parameter(W[:, 0], requires_grad=False)
    weight = nn.Parameter(W[:, 1:], requires_grad=False)
    print('Global LSM: {} \t L2 regular: {}'.format(
        tag, 'True' if reg_lambda != 0 else 'False'))
    return weight, bias


def io_check(fc_io, hidden, x):
    if fc_io:
        Hidden = torch.cat((torch.flatten(x, start_dim=1), hidden), dim=1)
        return Hidden
    else:
        return hidden

class PreTuner(Opt):
    def __init__(self,):
        super().__init__()
        
        self.arch_choice = ['cnn', 'esn']
        
    def tuning(self,hyper, data_pack, Deep, device):
        arch = Opt()
        arch.loss = float('inf')
        
        for arch_name in self.arch_choice:
            hyper.dict[arch_name].hyper.device = device            
            
            tuner_dir = os.path.join(data_pack.task_dir, 'tuner', 'Deep{}'.format(
            len(Deep)), 'Arch.{}'.format(arch_name))
            analysis_dir = os.path.join(tuner_dir,'Bayes_Search')
            if os.path.exists(analysis_dir):
                analysis = Analysis(analysis_dir)
                best_config = analysis.get_best_config(metric='e_norm', mode='min')
                
                df = analysis.dataframe(metric='e_norm', mode='min')
                arch_loss = df['e_norm'].min()
            else:
                atm = getattr(current_module, '{}Tuner'.format(arch_name))
                arch_tuner = atm(hyper.dict[arch_name], data_pack, Deep) 
                arch_tuner.conduct()
                arch_loss = arch_tuner.best_result['e_norm']
                best_config = arch_tuner.best_config
            
            if arch_loss <= arch.loss:
                arch.best_config = best_config
                arch.name = arch_name
                arch.loss = arch_loss

        atm = getattr(current_module, '{}Tuner'.format(arch.name))
        updater = atm(hyper.dict[arch.name], data_pack, Deep) 
        updater.update(arch.best_config)
            
        arch.best_arch = updater.best_arch
        arch.ho = updater.ho
        arch.fc_io = updater.fc_io
        arch.e_norm = updater.e_norm
        arch.device = device
        
        return arch, updater.data

class ArchTuner(Opt):
    def __init__(self, arch_opts, data_pack, Deep):
        super().__init__()

        self.merge(arch_opts)
        self.device = self.hyper.device

        self.data_init(data_pack, Deep)

        # self.tuner.metirc = 'e_norm'
        self.metric = 'e_norm'
        if 'iters' not in self.tuning.dict:
            self.tuner.num_samples = 16
        else:
            self.tuner.num_samples = self.tuning.iters

        self.cpus = 30
        self.resource = {
            "cpu": 10,
            "gpu": 1  # set this for GPUs
        }
                
        # self.tuner.ho_epochs = 500

        self.arch_name = arch_opts.name

        Arch_dict = {
            'esn': esnLayer,
            'cnn': cnnLayer,
            # 'ces': cesLayer,
            # 'esc': escLayer,
        }

        self.arch_func = Arch_dict[self.arch_name]

        self.loss_fn = nn.MSELoss()
        self.tuner_dir = os.path.join(self.data.task_dir, 'tuner', 'Deep{}'.format(
            self.DeepNum), 'Arch.{}'.format(self.arch_name))
        if not os.path.exists(self.tuner_dir):
            os.makedirs(self.tuner_dir)

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

    def conduct(self,):
        ray.init(num_cpus=self.cpus, _redis_max_memory=10**9)
        # self.tuner.num_samples = 80
        algo = self.search_ax()

        cat_e = torch.cat(self.data.e, dim=0)
        cat_x = torch.cat(self.data.x, dim=0) 

        cell_data = Opt()
        cell_data.d_fx = self.data.fx
        cell_data.d_cate = cat_e
        cell_data.d_catx = cat_x

        cell_data.merge(self, ['hyper', 'arch_name',
                        'device', 'steps', 'input_dim', 'H', 'xm_size'])

        analysis = tune.run(
            tune.with_parameters(tuningCell, data=cell_data),
            name=self.tuner.name,
            search_alg=algo,
            config=self.tuning.dict,
            resources_per_trial=self.resource,
            metric=self.metric,
            mode="min",
            num_samples=self.tuner.num_samples,
            local_dir=self.tuner_dir,
            verbose=1,
            stop={
                "training_iteration": self.tuner.num_samples
            }
        )

        self.best_config = analysis.get_best_config()
        self.best_result = analysis.best_result
        ray.shutdown()

    def deep_forward(self, input):
        sum_p = torch.zeros((input.shape[0], self.data.H)).to(self.device)
        if len(self.Deep) == 0:
            return input, sum_p
        else:
            input_x = input.detach().clone()
            for stack in self.Deep:
                input, _, p = stack(input, input_x)
                sum_p += p
            return input, sum_p

    def data_init(self, data_pack, Deep):
        self.merge(data_pack, ['task_dir', 'steps', 'input_dim', 'H', 'xm_size'])
        
        data = Opt()
        data.merge(data_pack, ['task_dir', 'steps', 'input_dim', 'H', 'xm_size'])

        with torch.no_grad():
            data.x = [_i.detach().clone() for _i in data_pack.x]
            data.y = [_i.detach().clone() for _i in data_pack.y]
            data.e = [_i.detach().clone() for _i in data_pack.e]

        self.Deep = Deep
        self.DeepNum = len(self.Deep)
        self.data = data

        self.data.fx = []
        self.data.sp = []

        with torch.no_grad():
            for _x in self.data.x:
                # _sp = torch.zeros((_x.shape[0], self.data.H)).to(self.device)
                _fx, _sp = self.deep_forward(_x)
                self.data.fx.append(_fx)
                self.data.sp.append(_sp)

    def get_map(self, _hyper, data=None):
        map_size = None
        return map_size

    def gen_ho(self, _hyper, fc_io):
        map_size = self.get_map(_hyper)
        if fc_io:
            map_size += self.xm_size
        ho = fcLayer(map_size, self.H, self.device)
        return ho

        # return analysis

        # print("Best config is:", analysis.best_config)

    def ho_update(self, model, ho, d_fx, d_catx, d_cate):
        cat_h = []
        for fx in d_fx:
            _, h = model(fx)
            cat_h.append(h)
        cat_h = torch.cat(cat_h, dim=0)
        cat_h = io_check(self.fc_io, cat_h, d_catx)
        weight, bias = close_ho(self.device, cat_h, d_cate)
        ho.update(weight, bias)
        torch.cuda.empty_cache() if next(ho.parameters()).is_cuda else gc.collect()
        return ho
    
    def data_update(self,):
        # e_list = []
        for i, (_fx, _sp) in enumerate(zip(self.data.fx, self.data.sp)):
            # print(torch.dist(_x, self.data.x[i]))
            _, fm_flatten = self.best_arch(_fx)
            fm_flatten = io_check(self.fc_io, fm_flatten, self.data.x[i])
            _p = self.ho(fm_flatten)
            sum_p = _sp + _p
            # e_list.append(_e)
            # self.data.p[i] = sum_p
            self.data.e[i] = self.data.y[i] - sum_p
            # self.data.x[i] = fm
            torch.cuda.empty_cache() if next(self.ho.parameters()).is_cuda else gc.collect()
        # cat_e = torch.cat(e_list, dim=0)
        # print("**** Last Error 2-norm: {}".format(torch.linalg.matrix_norm(cat_e).item()))
        new_cat_e = torch.cat(self.data.e, dim=0)
        self.e_norm = torch.linalg.matrix_norm(new_cat_e).item()
        # print("**** Curr Error 2-norm: {}".format(self.e_norm))

    def dfx_update(self,):
        """Updating data feature of deep forwarded input X."""
        pass

    def update(self, best_config):
        self.fc_io = best_config['fc_io']        
        self.hyper.input_dim = self.data.input_dim
        best_config.pop('fc_io')
        self.hyper.update(best_config)        
        self.best_arch = self.arch_func(**self.hyper.dict).to(self.device)
        self.best_arch.freeze()

        ho = self.gen_ho(self.hyper, self.fc_io)
        ho.freeze()

        cat_e = torch.cat(self.data.e, dim=0)
        cat_x = torch.cat(self.data.x, dim=0)
        self.ho = self.ho_update(
            self.best_arch, ho, d_fx=self.data.fx, d_cate=cat_e, d_catx=cat_x)
        self.data_update()
        self.dfx_update()
        
class tuningCell(tune.Trainable):
    def setup(self, config, data=None):
        # tune.utils.wait_for_gpu(target_util = 0.8) # need pip install gputil
        self.d_fx = data.d_fx
        self.d_cate = data.d_cate
        self.d_catx = data.d_catx

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

        Arch_dict = {
            'esn': esnLayer,
            'cnn': cnnLayer,
        }
        
        _hyper.dict.pop('fc_io')
        self.model = Arch_dict[self.arch_name](**_hyper.dict).to(self.device)
        self.model.freeze()
        self.ho = self.gen_ho(_hyper)
        self.ho.freeze()

        self.loss_fn = nn.MSELoss()

    def gen_ho(self, _hyper):
        if self.arch_name == 'cnn':
            map_size = self.cnn_map(_hyper)
        elif self.arch_name == 'esn':
            map_size = self.rnn_map(_hyper)
        else:
            raise ValueError('Non-supported arch type: {}'.format(self.arch_name)
                             )

        if self.fc_io:
            map_size += self.xm_size
        ho = fcLayer(map_size, self.H, self.device)
        return ho

    def cnn_map(self, _hyper):
        pooling_redcution = 0
        if _hyper.pooling:
            pooling_redcution = _hyper.pooling_size - 1
        conv_redcution = 0
        if _hyper.padding != 'same':
            conv_redcution = _hyper.kernel_size - 1

        map_size = self.steps - conv_redcution - pooling_redcution
        return map_size

    def rnn_map(self, _hyper):
        map_size = _hyper.hidden_size
        return map_size

    def step(self,):
        cat_h = []
        for fx in self.d_fx:
            _, h = self.model(fx)
            cat_h.append(h)
        cat_h = torch.cat(cat_h, dim=0)

        cat_h = io_check(self.fc_io, cat_h, self.d_catx)

        weight, bias = close_ho(self.device, cat_h, self.d_cate)
        self.ho.update(weight, bias)
        cat_p = self.ho(cat_h)
        # loss = self.loss_fn(cat_p, self.d_cate).item()

        cat_e = self.d_cate - cat_p
        e_norm = torch.linalg.matrix_norm(cat_e).item()

        return {
            # "loss": loss,
            "e_norm": e_norm
        }


class esnTuner(ArchTuner):
    def __init__(self, arch_opts, data_pack, Deep):
        super().__init__(arch_opts, data_pack, Deep)

    def get_map(self, _hyper, data=None):
        map_size = _hyper.hidden_size
        return map_size

    def dfx_update(self):
        self.data.input_dim = self.hyper.hidden_size


class cnnTuner(ArchTuner):
    def __init__(self, arch_opts, data_pack, Deep):
        super().__init__(arch_opts, data_pack, Deep)

    def get_map(self, _hyper,):
        pooling_redcution = 0
        if _hyper.pooling:
            pooling_redcution = _hyper.pooling_size - 1
        conv_redcution = 0
        if _hyper.padding != 'same':
            conv_redcution = _hyper.kernel_size - 1

        map_size = self.steps - conv_redcution - pooling_redcution
        return map_size

    def dfx_update(self,):
        self.data.input_dim = 1

        pooling_redcution = 0
        if self.hyper.pooling:
            pooling_redcution = self.hyper.pooling_size - 1
        conv_redcution = 0
        if self.hyper.padding != 'same':
            conv_redcution = self.hyper.kernel_size - 1

        map_size = self.data.steps - conv_redcution - pooling_redcution
        print('***** current map size: {}'.format(map_size))
        self.data.steps = map_size
