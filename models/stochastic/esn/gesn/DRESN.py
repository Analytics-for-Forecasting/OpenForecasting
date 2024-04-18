# @article{yangDynamical2019,
#   title = {Dynamical Regularized Echo State Network for Time Series Prediction},
#   author = {Yang, Cuili and Qiao, Junfei and Wang, Lei and Zhu, Xinxin},
#   year = {2019},
#   journal = {Neural Computing and Applications},
#   volume = {31},
#   number = {10},
#   pages = {6781--6794}
# }

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskLoader import Opt

import torch
import torch.nn as nn
import numpy as np

from models.stochastic.Base import esnLayer
from models.stochastic.esn.gesn.GrowingESN import Growing_ESN
from tqdm import trange,tqdm
import math
import copy,gc

class DRESN(Growing_ESN):
    def __init__(self, opts, logger):
        opts.hidden_size = opts.Ng1
        super().__init__(opts=opts, logger=logger)
        
        self.Ng1=opts.Ng1
        self.Ng2=opts.Ng2
        self.Total_Hidden_Size = self.Hidden_Size 
        self.opts.nonlinearity = 'sigmoid'
        
    def init_arch(self):
        self.res_list = nn.ModuleList([])
        self.ho_list = nn.ModuleList([])
        
        self.best_res_list = None
        self.best_ho_list = None
        
    def log_epoch(self,min_vrmse, loss, vloss):
        self.logger.info('Best VRMSE: {:.8f} \t  Training RMSE: {:.8f} \t Validating RMSE: {:.8f}'.format(
            min_vrmse, loss, vloss))
        
        self.fit_info.trmse = self.fit_info.loss_list[-1]
        self.fit_info.vrmse = min_vrmse   
    
    def update_readout(self, Hidden, y, out_layer = None):
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
    
    def xfit(self, train_data, val_data):
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        
        min_vrmse = float('inf')

        
        
        # ----init hidden layer: Step 1----
        subs = 1
        self.logger.info('-'*55)
        self.logger.info(
            'Iterations: {} \t Reservoir size: {}'.format(
                subs,self.Total_Hidden_Size))
        cur_hidden_size = self.Hidden_Size
        layer_esn = esnLayer(
            init='svd',
            hidden_size=cur_hidden_size,
            input_dim=self.Input_dim,
            nonlinearity=self.opts.nonlinearity,
            leaky_r=self.opts.leaky_r,
            weight_scale=self.opts.weight_scaling,
            iw_bound=self.opts.iw_bound,
            hw_bound=self.opts.hw_bound,
            device=self.device
        )
        layer_esn.freeze()
        # cur_readout = nn.Linear(cur_hidden_size, self.Output_dim)
        
        H_state, x, y = self.batch_transform(train_loader, layer_esn)
        
        lossFn_y = y[:,:, -1].cpu()
        lossFn_hstate = H_state[:,:, -1]
        
        H_state = self.stripS_process(H_state)
        y = self.stripS_process(y)
        Error = y.detach().clone()
        
        
        cur_readout = self.update_readout(H_state, Error)
        init_pred = cur_readout(H_state)
        
        Error = Error - init_pred
        
        lossFn_pred = cur_readout(lossFn_hstate)
        
        loss = np.sqrt(self.loss_fn(lossFn_pred, lossFn_y).item())
        self.fit_info.loss_list.append(loss)
        
        
        vh_state, val_x, val_y = self.batch_transform(val_loader, layer_esn)
        lossFn_val_y = val_y[:,:,-1].cpu()
        lossFn_vh_state = vh_state[:,:, -1]
        
        vloss = np.sqrt(self.loss_fn(cur_readout(lossFn_vh_state), lossFn_val_y).item())
        self.fit_info.vloss_list.append(vloss)
        
        self.res_list.append(layer_esn)
        self.ho_list.append(cur_readout)
        
        if vloss < min_vrmse:
            min_vrmse = vloss
            self.best_res_list = copy.deepcopy(self.res_list)
            self.best_ho_list = copy.deepcopy(self.ho_list)
            self.logger.info('****** Found new best state ******')
            
        self.log_epoch(min_vrmse, loss, vloss)
         
         #---------------------------------------------
        for i in trange(1, self.Subreservoir_Size):
            # ----try to replace the old arch: step 2
            q = math.ceil(self.Total_Hidden_Size / self.Ng1)
            
            Ng1_size = None
            Ng1_readout = None
            
            for p in trange(1,q+1):
                Ng1_res = esnLayer(
                    init='svd',
                    hidden_size= p * self.Ng1,
                    input_dim=self.Input_dim,
                    nonlinearity=self.opts.nonlinearity,
                    leaky_r=self.opts.leaky_r,
                    weight_scale=self.opts.weight_scaling,
                    iw_bound=self.opts.iw_bound,
                    hw_bound=self.opts.hw_bound,
                    device=self.device
                )
                Ng1_res.freeze()
                H_state, x, y = self.batch_transform(train_loader, Ng1_res)
                lossFn_Ng1_hstate = H_state[:,:,-1]
                H_state = self.stripS_process(H_state)
                y = self.stripS_process(y)                
                Ng1_readout =  self.update_readout(H_state, y)    
                
                Ng1_pred =  Ng1_readout(H_state)
                
                Error_Ng1 = y - Ng1_pred
                
                Enorm_Ng1 = torch.linalg.matrix_norm(Error_Ng1).item()
                Enorm_base = torch.linalg.matrix_norm(Error).item()
                
                if Enorm_Ng1 <= Enorm_base:
                    Ng1_size = p * self.Ng1
                    lossFn_pred = Ng1_readout(lossFn_Ng1_hstate)
                    
                    self.res_list = nn.ModuleList([Ng1_res])
                    self.ho_list = nn.ModuleList([Ng1_readout])
                    self.Total_Hidden_Size = Ng1_size
                    Error = Error_Ng1
                    self.logger.info(
                    'Find new Ng1 to replace the current hidden layer')
                    break
            
            # ----Generate Ng2 arch to grow the hidden layer: Step 3----
            Ng2_res = esnLayer(
                init='svd',
                hidden_size= self.Ng2,
                input_dim=self.Input_dim,
                nonlinearity=self.opts.nonlinearity,
                leaky_r=self.opts.leaky_r,
                weight_scale=self.opts.weight_scaling,
                iw_bound=self.opts.iw_bound,
                hw_bound=self.opts.hw_bound,
                device=self.device
            )
            Ng2_res.freeze()
            H_state, _, _ = self.batch_transform(train_loader, Ng2_res)
            lossFn_Ng2_hstate = H_state[:,:, -1]
            
            H_state = self.stripS_process(H_state)
            Ng2_readout =  self.update_readout(H_state, Error)
            Ng2_pred = Ng2_readout(H_state)
            Error = Error - Ng2_pred
            
            self.res_list.append(Ng2_res)
            self.ho_list.append(Ng2_readout)
            
            self.Total_Hidden_Size += self.Ng2
            subs = i + 1
            self.logger.info('-'*55)
            self.logger.info(
                'Iterations: {} \t Reservoir size: {}'.format(
                    subs,self.Total_Hidden_Size))

            lossFn_Ng2_pred = Ng2_readout(lossFn_Ng2_hstate)
            lossFn_pred = lossFn_pred + lossFn_Ng2_pred
            
            lossFn_vpred = self.forward(val_x, self.res_list, self.ho_list)
            
            loss = np.sqrt(self.loss_fn(lossFn_pred, lossFn_y).item())
            vloss = np.sqrt(self.loss_fn(lossFn_vpred, lossFn_val_y).item())
            
            self.fit_info.loss_list.append(loss)
            self.fit_info.vloss_list.append(vloss)
            torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
            
            if vloss < min_vrmse:
                min_vrmse = vloss
                self.best_res_list = copy.deepcopy(self.res_list)
                self.best_ho_list = copy.deepcopy(self.ho_list)
                self.logger.info('****** Found new best state ******')
                
            self.log_epoch(min_vrmse, loss, vloss)

        return self.fit_info
            
    def forward(self, x, res_list = None, ho_list = None):
        
        if res_list is None:
            res_list = self.best_res_list
        if ho_list is None:
            ho_list = self.best_ho_list
        
        samples = x.shape[0]
        pred = torch.zeros((samples, self.Output_dim))
        for res, ho in zip(res_list, ho_list):
            _, h_state = res(x)
            h_state=h_state.to(self._device)
            _pred = ho(h_state)
            pred = pred  + _pred
        return pred

    
