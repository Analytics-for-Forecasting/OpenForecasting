import os
from re import S
import sys

from numpy.lib.function_base import select

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import math
import gc
from tqdm import trange,tqdm

import torch
import torch.nn as nn
import numpy as np

# from models.stochastic.Evolve import ReadoutTuner
from task.TaskLoader import Opt
# from models.stochastic.cnn.ice.tuner_demo import Arch_dict, ArchTuner

from models.stochastic.cnn.ice.deeper.deeper_tuner import PreTuner
import importlib

# def patience_check(L):
#     return all(x<=y for x, y in zip(L, L[1:]))

class stack(nn.Module):
    def __init__(self, arc_obj):
        super(stack, self).__init__()
        
        self.arch = arc_obj.best_arch
        self.ho = arc_obj.ho
        self.fc_io = arc_obj.fc_io
        
        # print(self)
        # self.batchNorm = torch.nn.BatchNorm1d(arch_tuner.data.input_dim, affine=False).to(arch_tuner.device)
    
    def io_check(self, hidden,x):
        if self.fc_io:
            Hidden = torch.cat((torch.flatten(x,start_dim=1), hidden),dim=1)
            return Hidden
        else:
            return hidden
    
    def forward(self, input, input_x):
        fm, fm_flatten = self.arch(input)
        fm_flatten = self.io_check(fm_flatten, input_x)
        pred = self.ho(fm_flatten)
        # fm = self.batchNorm(fm)
        return fm, fm_flatten, pred

class ICESN(nn.Module):
    """
    From Deep ESN to Adaptive Convolutional ESN: An adaptive method to incrementally craft convolutional ESN for time series forecasting.
    """
    def __init__(self, opts, logger):
        super(ICESN, self).__init__()
        self.hyper = opts
        self.patience = 5
        self.patience_bos = 10 #the constructive process should early-stop after the depth > patience_bos and the valid-loss does not small (aka. improve) than the valid-loss min(list[-patience:] )
        self.logger = logger
        self.device = self.hyper.device    

        self.arch_choice = ['esn','cnn']
        # self.arch_choice = ['cnn']
        
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
            
        if self.device == torch.device('cpu'):
            self.logger.info('Using CPU...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.device)
        # self.branch_size = opts.branch_size # the largest branches of the finally neural architectures
        # self.best_size = 0
        # In the objection, the branch list should be a 2D tree, which stores the depth and the width of the neural architecture.
        # branch: diversing branch
        # to clear
        # bough: main branch
        
        # during the first trial version, the max width is set as 1, the arch. grows deeper.
        # self.pointerTree = [] # to do, store the list of the pointers which point to the afore-arch from the current arch in the branch list.        
        self.mse = nn.MSELoss()
        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        # self.deepth = 0
        self.max_deepth = 20
        self.best_depth = None
        # self.width = 0
        # self.max_width = 20
        self.init_arch()
        
        self.preTuner_path = 'models.stochastic.cnn.ice.deeper.deeper_tuner'
    
    
    def patience_check(self):
        pat_Tag = True
        curr_depth = len(self.fit_info.vloss_list)
        if self.patience_bos <= self.patience:
            self.patience_bos = self.patience
        if curr_depth > self.patience_bos:
            min_vl = min(self.fit_info.vloss_list[-self.patience:])
            last_vl = self.fit_info.vloss_list[-1]
            if last_vl > min_vl:
                pat_Tag = False
        
        return pat_Tag
      
    def init_arch(self,):
        # Deep: main branch; Width: extra branch
        # self.Width = nn.ModuleList([])
        self.Deep = nn.ModuleList([])
            
    def solve_output(self, Hidden_States, y, reg_lambda=1.0):
        t_hs = torch.cat((torch.ones(Hidden_States.size(0), 1).to(
            self.device), Hidden_States), dim=1)
        HTH = t_hs.T @ t_hs
        HTY = t_hs.T @ y
        # ridge regression with pytorch
        I = (reg_lambda * torch.eye(HTH.size(0))).to(self.device)
        A = HTH + I
        # orig_rank = torch.matrix_rank(HTH).item() # torch.matrix_rank is deprecated in favor of torch.linalg.matrix_rank in v1.9.0
        orig_rank = torch.linalg.matrix_rank(A, hermitian=True).item()
        tag = 'Inverse' if orig_rank == t_hs.size(1) else 'Pseudo-inverse'
        if tag == 'Inverse':
            W = torch.mm(torch.inverse(A), HTY).t()
        else:
            W = torch.mm(torch.linalg.pinv(A.cpu()),
                         HTY.cpu()).t().to(self.device)
        return W, tag
    
    def update_readout(self, Hidden,  y, reg_lambda=1.0, show = False):
        # _Hidden = self.io_check(Hidden, x)
        W, tag = self.solve_output(Hidden, y, reg_lambda)
        if show:
            self.logger.info('LSM: {} \t L2 regular: {}'.format(            tag, 'True' if self.hyper.reg_lambda != 0 else 'False'))
        ho = nn.Linear(Hidden.size(1), y.size(1))
        ho.bias = nn.Parameter(W[:, 0], requires_grad=False)
        ho.weight = nn.Parameter(W[:, 1:], requires_grad=False)
        ho.inverse = tag
        return ho
        

    def loaderPack(self, data_loader):
        data_pack = Opt()
        data_pack.task_dir = self.hyper.task_dir
        # data_pack.fc_io = self.fc_io
        data_pack.x = []
        data_pack.y = []
        data_pack.e = []
        for _x, _y in (data_loader):
            _x = _x.to(self.device)
            _y = _y.to(self.device)
            data_pack.x.append(_x)
            data_pack.y.append(_y)
            _p = torch.zeros_like(_y).to(self.device)
            _e = _y.detach().clone()
            data_pack.e.append(_e)
            
            data_pack.steps = _x.shape[2]
            data_pack.input_dim = _x.shape[1]
            data_pack.H = _y.shape[1]
            data_pack.xm_size = _x.shape[1] * _x.shape[2]

        return data_pack
    
    
    def forward(self, input, deepNum = None):
        sum_pred = torch.zeros(
            (input.data.size(0), self.hyper.H)).to(self.device)
        
        input_x = input.detach().clone()
        
        _deepNum = deepNum if deepNum is not None else len(self.Deep)
        
        for stack in self.Deep[:_deepNum]:
            input, _, pred = stack(input,input_x)
            sum_pred += pred

        return sum_pred
    
    def xfit(self, train_loader, val_loader):
        '''
        '''
        min_vrmse = float('inf')        
        data_pack = self.loaderPack(train_loader)
            
        for depth in trange(self.max_deepth):
            patience_pass = self.patience_check()
            if patience_pass == False:
                break
            
            preT = PreTuner()
            arch_delta, data_pack = preT.tuning(self.hyper, data_pack,self.Deep, self.device)
            
            # arches need conclude data_pack
            _stack = stack(arch_delta)
            self.logger.info(_stack)
            self.Deep.append(_stack)
            # data_pack = arch_delta.data
            torch.cuda.empty_cache() if next(_stack.parameters()).is_cuda else gc.collect()
            
            
            cat_y = torch.cat(data_pack.y, dim=0)
            cat_p = cat_y - torch.cat(data_pack.e, dim=0)
            rmse = np.sqrt(self.mse(cat_p, cat_y).item())
            self.fit_info.loss_list.append(rmse)
            
            vp = []
            vy = []
            
            for v_x, v_y in val_loader:
                v_x = v_x.to(self.device)
                v_y = v_y.to(self.device)
                vy.append(v_y)
                v_p = self.forward(v_x)
                vp.append(v_p)
            
            vp = torch.cat(vp, dim=0)
            vy = torch.cat(vy, dim=0)
            vrmse = np.sqrt(self.mse(vp, vy).item())
            self.fit_info.vloss_list.append(vrmse)
            
            if vrmse < min_vrmse:
                min_vrmse = vrmse
                self.best_depth = depth
                self.logger.info('Found new best state')
                self.logger.info('Best vrmse: {:.4f}'.format(min_vrmse))
            
            self.logger.info('Depth size: {} \t Arch type: {} \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f} \t Best VRMSE: {:.8f}'.format(
                depth+1, arch_delta.name, rmse, vrmse, min_vrmse))
            self.logger.info("**** Curr Error 2-norm: {}".format(arch_delta.e_norm))
            self.logger.info('>'*36)
            
        self.fit_info.trmse = self.fit_info.loss_list[self.best_depth]
        self.fit_info.vrmse = min_vrmse
        
        return self.fit_info

    def loader_pred(self, data_loader):
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred = self.forward(batch_x, deepNum=self.best_depth + 1)
            x.append(batch_x)
            y.append(batch_y)
            pred.append(batch_pred)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()
        
        return x, y, pred
    # def loss_fn(self, Hidden, Hidden_delta, E):
    #     '''To Do'''
    #     pass
        
    
    # def supervised_loss(self, state, E, reg_lambda=0):
    #     W,_ = self.solve_output(state, E, reg_lambda=reg_lambda)
    #     ho = nn.Linear(state.shape[1], E.shape[1])
    #     ho.bias = nn.Parameter(W[:, 0], requires_grad=False)
    #     ho.weight = nn.Parameter(W[:, 1:], requires_grad=False)
    #     P = ho(state)
    #     loss = self.mse(P, E).item()
    #     return ho, P, loss

    # def preTrainBranch(self,sub,E):
    #     '''to do'''
    #     pass
