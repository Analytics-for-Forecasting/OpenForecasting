# @article{liPSObased2019,
#   title = {{{PSO-based}} Growing Echo State Network},
#   author = {Li, Ying and Li, Fanjun},
#   year = {2019},
#   journal = {Applied Soft Computing},
#   volume = {85},
#   pages = {105774}
# }

import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.path.pardir, os.path.pardir))

import gc
import copy
from tqdm import trange, tqdm
import ray
import nevergrad as ng
from ray.tune.search.nevergrad import NevergradSearch
from ray.air.config import CheckpointConfig
from ray.air.config import RunConfig
from ray.air import session, FailureConfig
from ray import tune
from task.util import set_logger
import math
from models.stochastic.esn.gesn.GrowingESN import Growing_ESN
from models.stochastic.Base import esnLayer
import numpy as np
import torch.nn as nn
import torch
from task.TaskLoader import Opt
from ray.tune import ExperimentAnalysis
from ray.tune.search import ConcurrencyLimiter

class PSOGESN(Growing_ESN):
    def __init__(self, opts, logger):
        super().__init__(opts=opts, logger=logger)

        self.pso_c1 = opts.pso_c1 if 'pso_c1' in opts.dict else 0.3
        self.pso_c2 = opts.pso_c2 if 'pso_c2' in opts.dict else 0.2
        self.pso_w = opts.pso_w if 'pso_w' in opts.dict else 0.2
        self.pso_cpu = opts.pso_cpu if 'pso_cpu' in opts.dict else 30
        self.pso_gpu = opts.pso_gpu if 'pso_gpu' in opts.dict else 1
        

        # pso_iters and pso_pops settings refer to Sec. 4 of the published paper: PSO-based growing echo state network
        self.pso_iters = opts.pso_iters if 'pso_iters' in opts.dict else 10
        self.pso_pops = opts.pso_pops if 'pso_pops' in opts.dict else 10

        self.host_dir = self.opts.model_fit_dir
        self.tuner_dir = os.path.join(self.host_dir, 'tuner', 'cid_{}'.format(self.opts.cid))

    def init_arch(self):
        self.res_list = nn.ModuleList([])
        self.ho_list = nn.ModuleList([])

    def forward(self, x, res_index=None):
        if res_index is None:
            res_index = self.best_res_idx

        samples = x.shape[0]

        pred = torch.zeros((samples, self.Output_dim))

        for i in range(res_index + 1):
            _, h_state = self.res_list[i](x)
            h_state = h_state.to(self._device)
            _pred = self.ho_list[i](h_state)
            pred = _pred + pred

        return pred

    def catData_gen(self, data_loader):
        x = []
        y = []
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            x.append(batch_x)
            y.append(batch_y)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        return x, y

    def update_readout(self, Hidden, y, out_layer=None):
        # Hidden = self.stripS_process(Hidden)
        # y = self.stripS_process(y)
        W, tag = self.solve_output(Hidden, y)
        if out_layer is None:
            hidden_dim = Hidden.size(1)
            out_layer = nn.Linear(hidden_dim, self.Output_dim)
            out_layer.bias = nn.Parameter(W[:, 0], requires_grad=False)
            out_layer.weight = nn.Parameter(W[:, 1:], requires_grad=False)
        else:
            assert out_layer.weight.data.size(1) == Hidden.size(1)
            out_layer.bias = nn.Parameter(W[:, 0], requires_grad=False)
            out_layer.weight = nn.Parameter(W[:, 1:], requires_grad=False)

        return out_layer

    def pso_subres(self, i, train_loader, Error):

        i_tuner_dir = os.path.join(self.tuner_dir, 'sub_{}'.format(i))
        if not os.path.exists(i_tuner_dir):
            os.makedirs(i_tuner_dir)

        # tuner_path = os.path.join(
        #     i_tuner_dir, 'diag.best.pt')
        
        best_checkpoint_path = None
        analysis_dir = os.path.join(i_tuner_dir, 'PSO_Search')
        if os.path.exists(analysis_dir):
            analysis = ExperimentAnalysis(analysis_dir)
            best_checkpoint = analysis.get_best_checkpoint(analysis.get_best_logdir(metric='mse', mode='min'), metric='mse', mode='min')
            best_checkpoint_path = os.path.join(best_checkpoint.path, 'model.pth')

            self.logger.info('****** Load pre-training sub_{} ******'.format(i))
        else:
            self.logger.info('****** Start pre-training sub_{} ******'.format(i))

            func_data = Opt()
            func_data.train_loader = train_loader
            func_data.Error = Error
            func_data.opts = Opt(self.opts)
            
            tuning_dict = {}
            for u_i in range(self.Hidden_Size):
                tuning_dict['sigV_{}'.format(u_i)] = tune.uniform(0.1, 0.99)
            
            # -------
            pso_search = ConcurrencyLimiter(NevergradSearch(
                optimizer=ng.optimizers.ConfiguredPSO(
                    popsize=self.pso_pops,
                    omega=self.pso_w,
                    phip=self.pso_c1,
                    phig=self.pso_c2
                ),
                metric='mse',
                mode="min"
            ),max_concurrent=4)
            
            ray.init(num_cpus=self.pso_cpu)
            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(sub_esn_ind, data=func_data),
                    resources={"cpu": math.floor(self.pso_cpu * self.pso_gpu), "gpu": self.pso_gpu} # Round numbers down to the max. integer <= x
                ),
                param_space=tuning_dict,
                tune_config=tune.TuneConfig(
                    search_alg=pso_search,
                    metric='mse',
                    mode="min",
                    num_samples=self.pso_pops * self.pso_iters,
                ),
                run_config=RunConfig(
                    name='PSO_Search',
                    storage_path=i_tuner_dir,
                    verbose=1,
                    failure_config=FailureConfig(
                        max_failures=self.pso_pops * self.pso_iters // 2),
                    # Beacasue of the stochastic mechanism, the weight training iteration of the stochastic model is only one.
                    stop={'training_iteration': 1},
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=1,
                        checkpoint_score_order = 'min',
                        checkpoint_at_end=True
                    ),
                    sync_config=tune.SyncConfig(
                        syncer=None
                    )
                )
            )
            results = tuner.fit()
            ray.shutdown()
            best_result = results.get_best_result('mse', 'min')
            # best_config = best_result.config
            best_checkpoint_path = os.path.join(best_result.checkpoint.path, 'model.pth')
            
            # -------
            
            # ---Unit Test---
            # test_config = {}
            # for key in tuning_dict:
            #     test_config[key] = tuning_dict[key].sample()
            # tuning = once_trail(test_config, func_data)
            # result = tuning.step()
            # best_config = test_config
            # singular_list = []
            # for key in best_config:
            #     singular_list.append(best_config[key])
            # svd_s = torch.tensor(singular_list)
            # torch.save(svd_s, tuner_path)
            # ---Unit Test---
            
            self.logger.info('****** End pre-training sub_{} ******'.format(i))
  
        
        sub_esn = esnLayer(
            init = 'svd',
            hidden_size=self.Hidden_Size,
            input_dim=self.Input_dim,
            nonlinearity=self.opts.nonlinearity,
            leaky_r=self.opts.leaky_r,
            weight_scale=self.opts.weight_scaling,
            iw_bound=self.opts.iw_bound,
            hw_bound=self.opts.hw_bound,
            device=self.device
        )
        
        if best_checkpoint_path is not None:
            try:
                # best_checkpoint_path = best_checkpoint.as_directory()

                best_state = torch.load(best_checkpoint_path)
                sub_esn.load_state_dict(best_state)
                self.logger.info('****** Passing pre-training sub_{} ******'.format(i))
            except:
                self.logger.info('****** Fail to Pass pre-training sub_{} ******'.format(i))
        else:
            self.logger.info('****** Passing naive sub_{} ******'.format(i))
        
        sub_esn.freeze()

        H_state, _, _ = self.batch_transform(train_loader, sub_esn)
        T_state = H_state[:,:, -1]
        
        H_state = self.stripS_process(H_state)
        sub_readout = self.update_readout(H_state, Error)
        # sub_readout.freeze()
        
        sub_pred = sub_readout(H_state)
        lossFn_pred = sub_readout(T_state)

        return copy.deepcopy(sub_esn), copy.deepcopy(sub_readout), sub_pred, lossFn_pred

    def xfit(self, train_data, val_data):
        with torch.no_grad():
            train_loader = self.data_loader(train_data)
            val_loader = self.data_loader(val_data)

            min_vrmse = float('inf')

            _, catY = self.catData_gen(train_loader)
            _, catvY = self.catData_gen(val_loader)

            lossFn_y = catY[:,:, -1].cpu()
            lossFn_pred = torch.zeros_like(lossFn_y)
            
            lossFn_val_y = catvY[:,:, -1].cpu()
            lossFn_val_pred = torch.zeros_like(lossFn_val_y)
            
            Error = self.stripS_process(catY).detach().clone()
        
            for i in trange(self.Subreservoir_Size):
                self.logger.info('-'*55)
                self.logger.info(
                    'Subs index: {} \t Reservoir size: {}'.format(
                        i, self.Hidden_Size * (i+1)))

                sub_esn, sub_readout, sub_pred, t_pred = self.pso_subres( i, train_loader, Error)
                self.res_list.append(sub_esn)
                self.ho_list.append(sub_readout)
                
                Error = Error - sub_pred
                lossFn_pred =lossFn_pred + t_pred
                
                loss = np.sqrt(self.loss_fn(lossFn_pred, lossFn_y).item())
                self.fit_info.loss_list.append(loss)
                
                vh_state, val_x, val_y = self.batch_transform(val_loader, sub_esn)
                lossFn_val_y = val_y[:,:,-1].cpu()
                lossFn_vh_state = vh_state[:,:, -1]
                
                lossFn_val_pred = lossFn_val_pred + sub_readout(lossFn_vh_state)
                vloss = np.sqrt(self.loss_fn(lossFn_val_pred, lossFn_val_y).item())
                self.fit_info.vloss_list.append(vloss)
                torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
                
                if vloss < min_vrmse:
                    min_vrmse = vloss
                    self.best_res_idx = i
                    self.logger.info('****** Found new best state ******')
                
                self.logger.info('Best VRMSE: {:.8f} \t Best Res_idx: {} \t  Training RMSE: {:.8f} \t Validating RMSE: {:.8f}'.format(
                    min_vrmse, self.best_res_idx, loss, vloss))
                
            self.fit_info.trmse = self.fit_info.loss_list[self.best_res_idx]
            self.fit_info.vrmse = min_vrmse
        
        return self.fit_info


class sub_esn_ind(tune.Trainable):
    def setup(self, config, data=None):
        self.train_loader = data.train_loader
        self.Error = data.Error
        self.opts = data.opts
        # self.Hidden_size = data.Hidden_size
        # self.Input_dim = data.Input_dim
        # self.Output_dim = data.Output_dim
        # self.device = data.dvi

        singular_list = []
        for key in config:
            singular_list.append(config[key])

        assert self.opts.hidden_size == len(singular_list)

        self.sub_esn = esnLayer(
            hidden_size=self.opts.hidden_size,
            input_dim=self.opts.input_dim,
            nonlinearity=self.opts.nonlinearity,
            leaky_r=self.opts.leaky_r,
            weight_scale=self.opts.weight_scaling,
            iw_bound=self.opts.iw_bound,
            hw_bound=self.opts.hw_bound,
            device=self.opts.device
        )

        svd_s = torch.tensor(singular_list)
        w_hh = self.sub_esn.esnCell.svd_init(singular_values=svd_s)
        self.sub_esn.esnCell.hh.weight = nn.Parameter(w_hh)
        self.sub_esn.freeze()

        self._device = torch.device('cpu')

        self.Time_steps = self.opts.lag_order  # attention!
        self.Readout_steps = self.Time_steps if self.opts.readout_steps > self.Time_steps else self.opts.readout_steps

        stripS_operator = np.zeros(self.Time_steps)
        stripS_operator[-self.Readout_steps:] = 1
        self.stripS_operator = stripS_operator.tolist()

        self.loss_fn = nn.MSELoss()

    def stripS_process(self, state):
        '''state shape: samples, dim, steps
        '''
        select = []
        read_operator = self.stripS_operator.copy()

        assert state.shape[2] == len(read_operator)

        for id, tag in enumerate(read_operator):
            if int(tag) == 1:
                select.append(state[:, :, id])

        if len(select) == 0:
            select = state[:, :, -1:]
        else:
            select = torch.stack(select, dim=2).to(self._device)

        select = select.permute(0, 2, 1)
        select = torch.flatten(select, start_dim=0, end_dim=1).to(self._device)
        return select

    def solve_output(self, Hidden_States, y):
        t_hs = torch.cat((torch.ones(Hidden_States.size(0), 1).to(
            self._device), Hidden_States), dim=1)
        HTH = t_hs.T @ t_hs
        HTY = t_hs.T @ y
        # ridge regression with pytorch
        # I = (self.opts.reg_lambda * torch.eye(HTH.size(0))).to(self._device)
        A = HTH
        orig_rank = torch.linalg.matrix_rank(A, hermitian=True).item()
        tag = 'Inverse' if orig_rank == t_hs.size(1) else 'Pseudo-inverse'
        if tag == 'Inverse':
            W = torch.mm(torch.inverse(A), HTY).t()
        else:
            try:
                W = torch.linalg.lstsq(
                    A.cpu(), HTY.cpu(), driver='gelsd').solution.T.to(self._device)
            except:
                W = torch.mm(torch.linalg.pinv(A.cpu()),
                             HTY.cpu()).t().to(self._device)
        return W, tag

    def update_readout(self, Hidden, y, out_layer=None):
        W, tag = self.solve_output(Hidden, y)
        if out_layer is None:
            hidden_dim = Hidden.size(1)
            out_layer = nn.Linear(hidden_dim, self.opts.H)
            out_layer.bias = nn.Parameter(W[:, 0], requires_grad=False)
            out_layer.weight = nn.Parameter(W[:, 1:], requires_grad=False)
        else:
            assert out_layer.weight.data.size(1) == Hidden.size(1)
            out_layer.bias = nn.Parameter(W[:, 0], requires_grad=False)
            out_layer.weight = nn.Parameter(W[:, 1:], requires_grad=False)

        return out_layer

    def step(self,):
        h_states = []
        for batch_x, _ in self.train_loader:
            batch_x = batch_x.to(self.opts.device)
            _h_states, _ = self.sub_esn(batch_x)
            _h_states = _h_states.to(self._device)

            h_states.append(_h_states)
            torch.cuda.empty_cache() if next(
                self.sub_esn.parameters()).is_cuda else gc.collect()

        h_states = torch.cat(h_states, dim=0)

        h_states = self.stripS_process(h_states)
        sub_readout = self.update_readout(h_states, self.Error)
        # sub_readout.freeze()
        sub_pred = sub_readout(h_states)

        mse = self.loss_fn(sub_pred, self.Error).item()

        return {'mse': mse}

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.sub_esn.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.sub_esn.load_state_dict(torch.load(checkpoint_path))

class once_trail(sub_esn_ind):
    def __init__(self, config = None, data = None):
        self.setup(config,data)