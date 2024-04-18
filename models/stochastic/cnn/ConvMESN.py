# @article{maConvolutional2021,
#   title = {Convolutional Multitimescale Echo State Network},
#   author = {Ma, Qianli and Chen, Enhuan and Lin, Zhenxi and Yan, Jiangyue and Yu, Zhiwen and Ng, Wing W. Y.},
#   year = {2021},
#   journal = {IEEE Transactions on Cybernetics},
#   volume = {51},
#   number = {3},
#   pages = {1613--1625}
# }



import os,sys

from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import numpy as np
import torch.nn as nn
import torch
from tqdm import trange
import gc
from models.stochastic.Base import ESNCell
import sys
from task.TaskLoader import Opt
from  task.TaskLoader import torch_dataloader, dnn_dataset
from tqdm import trange,tqdm
import copy

class ConvMESN(nn.Module):
    def __init__(self, opts=None, logger=None):
        super(ConvMESN, self).__init__()
        self.opts = opts
        self.logger = logger
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
            
        self.Lag_order  = opts.lag_order
        self.Output_dim = opts.H
        self.Input_dim = opts.input_dim
        
        self.Hidden_Size = opts.hidden_size
        
        self.skip_lengths = opts.skip_lengths if 'skip_lengths' in opts.dict else [1,3,9,27]
        
        self.filter_types = opts.filter_types if 'filter_types' in opts.dict else 3
        
        self.filter_channels = opts.H * 2
        self.filter_heights = self.skip_lengths
        self.filter_width = self.Hidden_Size
        self.learning_rate = opts.learning_rate if 'learning_rate' in opts.dict else 1e-3
        self.step_lr = opts.step_lr if 'step_lr' in opts.dict else 3
        self.training_epochs = opts.training_epochs if 'training_epochs' in opts.dict else 30
        
        if max(self.skip_lengths) >=  self.Lag_order:
            raise ValueError('The max. skip_lengths [{}] is larger than the lag order [{}] of the input series'.format(max(self.skip_lengths), self.Lag_order))
        
        self.init_arch()

        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
                
        if self.opts.device == torch.device('cpu'):
            self.logger.info('Using CPU...')
            self.usingCUDA = False
        else:
            self.logger.info('Using Cuda...')
            self.to(self.opts.device)
            self.usingCUDA = True
        
        
    def init_arch(self,):
        '''Generate the reservoirs and check the max output height of the constructed filters.\n
        In this work, filter padding = valid, i.e., no padding, dilation = 1, and stride = 1.\n
        Note that: \n
        Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where `H_{out} = (H_{in}  + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1`
        
            
        '''
        self.num_reservoirs = len(self.skip_lengths)
        self.esns = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        self.leaky_r = self.opts.leaky_r
    
        max_height_config = max(self.filter_heights) * self.filter_types
        min_height = (self.Lag_order + 2 * 0 - 1 * (max_height_config - 1) - 1)/1 + 1
        
        if min_height <= 1:
            raise ValueError('Inappropriate filter_height and skip_length: {} in hyper.skip_lengths, which makes the output height of the filters be less than 2, making non convolutional operations for skip_length: {}'.format(max(self.skip_lengths),max(self.skip_lengths)))
        
        
        for i in range(self.num_reservoirs):
            
            esn = ESNCell(
                init='sparse', 
                hidden_size=self.Hidden_Size, 
                input_dim= self.Input_dim,
                nonlinearity='tanh',
                weight_scale=self.opts.weight_scaling,
                iw_bound=self.opts.iw_bound,
                sparsity=self.opts.sparsity,
                device=self.opts.device).float()
            esn.freeze()
            self.esns.append(esn)
            
            i_convs = nn.ModuleList([])
            for j in range(self.filter_types):
                conv = nn.Conv2d(1, self.filter_channels, (self.filter_heights[i] * (j + 1), self.filter_width),padding='valid').float()
                i_convs.append(conv)
            
            self.convs.append(i_convs)
        
        self.pooling = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc = nn.Linear(self.num_reservoirs * self.filter_types * self.filter_channels, self.Output_dim).float()
        
        weights = []
        weights.extend(self.convs.parameters())
        weights.extend(self.fc.parameters())
        self.optimizer = torch.optim.Adam(weights, lr=self.learning_rate)
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, self.step_lr, gamma=0.99)
        
        self.loss_fn = nn.MSELoss()
            
    def data_loader(self, data, _batch_size = None):
        '''
        Transform the numpy array data into the pytorch data_loader
        '''
        data_batch_size = self.opts.batch_size if _batch_size is None else _batch_size
        set_data = dnn_dataset(data, self.Output_dim, self.Lag_order,self.Input_dim)
        # set_if_x_data = if_choose(set_data.data,self.if_operator)
        # set_data.data = set_if_x_data
        
        set_loader = torch_dataloader(set_data, batch_size= data_batch_size,cuda= self.usingCUDA)
        return set_loader
        
    def get_echo_states(self, inputs, esn_index):
        '''return echo_states with shape (samples,time_steps, Hidden_Size)'''
        samples, time_steps = inputs.shape[0], inputs.shape[2]
        
        skip_length = self.skip_lengths[esn_index]
        
        x = [torch.zeros((samples,self.Hidden_Size)).to(self.opts.device) for n in range(skip_length)]
        
        echo_states = torch.empty((samples,time_steps, self.Hidden_Size)).to(self.opts.device)
        
        for t in range(time_steps):
            
            u = inputs[:,:,t]
            index =  t % skip_length
            
            xUpd = self.esns[esn_index](u, x[index])
            
            x[index] = (1-self.leaky_r) * x[index] + self.leaky_r * xUpd
            
            echo_states[:, t,:] = x[index]

        return echo_states
    

    def forward(self, batch_x):
        '''toDo: addding conv arch.'''
        multi_pools = []
        
        time_steps = batch_x.size(2)
        assert time_steps == self.Lag_order
        
        for i in range(self.num_reservoirs):
            with torch.no_grad():
                i_echo_states = self.get_echo_states(batch_x, i)
                # shape (samples,time_steps, Hidden_Size)
                i_echo_states = torch.unsqueeze(i_echo_states, dim=1)
                
            for j in range(self.filter_types):
                fm = self.convs[i][j](i_echo_states)
                fm = torch.relu(fm)
                fm = self.pooling(fm)
                fm = fm.squeeze(3).squeeze(2)
                multi_pools.append(fm)
                
        multi_pools = torch.cat(multi_pools, dim=1)
        pred = self.fc(multi_pools)
        
        return pred
        
    
    def xfit(self, train_data, val_data):
        # min_vmse = 9999
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)            
        
        min_vrmse = float('inf')
        
        self.best_convs = None
        self.best_fc = None
        self.best_epoch = 0
        
        for epoch in trange(self.training_epochs):
            
            rmse_train = 0
            for batch_x, batch_y in tqdm(train_loader):
                batch_x = batch_x.to(self.opts.device)
                batch_y = batch_y.to(self.opts.device)
                
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                loss = self.loss_fn(y_pred, batch_y)
                loss.backward()
                rmse_train += np.sqrt(loss.item())
                self.optimizer.step()
            
            rmse_train = rmse_train / len(train_loader)
            self.fit_info.loss_list.append(rmse_train)
            

            with torch.no_grad():
                rmse_val = 0
                for batch_x, batch_y in tqdm(val_loader):
                    batch_x = batch_x.to(self.opts.device)
                    batch_y = batch_y.to(self.opts.device)
                    y_pred = self(batch_x)
                    vloss = self.loss_fn(y_pred, batch_y)

                    rmse_val += np.sqrt(vloss.item())
                
                rmse_val = rmse_val / len(val_loader)
            self.fit_info.vloss_list.append(rmse_val)
            
            if rmse_val < min_vrmse:
                min_vrmse = rmse_val
                self.best_epoch = epoch
                self.logger.info('Found new best state')
                self.logger.info('Best vmse: {:.4f}'.format(min_vrmse))
                self.best_convs = copy.deepcopy(self.convs)
                self.best_fc = copy.deepcopy(self.fc)
                
            self.logger.info(
                'Epoch {}/{} \t \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f} \t Best VRMSE: {:.8f} \t Best Epoch: {}'.format(epoch + 1, self.training_epochs, rmse_train, rmse_val, min_vrmse, self.best_epoch))
            
        return self.fit_info
    
    def task_pred(self, task_data):
        data_loader = self.data_loader(task_data)
        x, y, pred = self.loader_pred(data_loader)
        
        return x, y, pred
    
    def loader_pred(self, data_loader, using_best = True):
        x = []
        y = []
        pred = []
        
        if using_best and self.best_convs is not None and self.fc is not None:
            self.convs = self.best_convs
            self.fc = self.best_fc
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(data_loader):
                batch_x = batch_x.to(self.opts.device)
                # batch_y = batch_y.to(self.opts.device)
                batch_pred = self.forward(batch_x)
                x.append(batch_x.cpu())
                y.append(batch_y.cpu())
                pred.append(batch_pred.cpu())
            
        x = torch.cat(x, dim=0).detach().numpy()
        y = torch.cat(y, dim=0).detach().numpy()
        pred = torch.cat(pred, dim=0).detach().numpy()
        
        return x, y, pred
            
            