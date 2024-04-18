import os
import sys
from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import numpy as np
import torch.nn as nn
import torch

from task.util import os_makedirs, os_rmdirs
from task.util import plot_xfit, savebest_checkpoint

from tqdm import trange,tqdm

import gc
from task.TaskLoader import torch_dataloader,mlp_dataset,Opt


def initWeight(input_dim, hidden_size=1, grad=False):
    result = torch.empty(input_dim, hidden_size).float()
    result = nn.init.normal_(result, std=0.1)
    if grad == True:
        result.requires_grad = True
    return result


def initBiases(hidden_size=1, grad=False):
    result = torch.empty(1, hidden_size).float()
    result = nn.init.normal_(result, std=0.1)
    if grad == True:
        result.requires_grad = True
    return result


class IELM(nn.Module):
    '''
    Incremental Extreme Learning Machine
    '''

    def __init__(self,opts=None, logger=None):
        self.opts = opts
        self.logger = logger
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        super(IELM, self).__init__()
        self.input_dim = opts.lag_order
        self.output_dim = opts.H

        self.best_hidden_size = 1
        self.hidden_size = opts.hidden_size if 'hidden_size' in opts.dict else 100
        self.Lambdas = opts.Lambdas if 'Lambdas' in opts.dict else [0.5, 1, 5 ,15, 30, 50, 100, 150, 200]
        self.tolerance = opts.tolerance if 'tol' in opts.dict else 0

        self.hidden = 1

        self.opts.device = torch.device('cpu')
        self.device = self.opts.device
        if self.opts.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
            self.usingCUDA = False
        else:
            self.logger.info('Using Cuda...')
            self.usingCUDA = True
            self.to(self.opts.device)
        
        with torch.no_grad():
            self.weight_IH = initWeight(self.input_dim, 1).to(self.opts.device)
            self.bias_IH = initBiases().to(self.opts.device)
        self.weight_HO = None
        self.weight_candidates = None
        self.bias_candidates = None


        self.loss_fn = nn.MSELoss()
        
        self.best_state = None
        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        
    
    def load_state(self,):
        assert self.best_state is not None
        self.hidden_size = self.best_state['hidden_size']
        self.best_hidden_size = self.best_state['best_hidden_size']
        self.weight_IH = self.best_state['weight_IH']
        self.bias_IH = self.best_state['bias_IH']
        self.weight_HO = self.best_state['weight_HO']
        self.best_HO = self.best_state['best_HO']


    def forward(self, input, size):
        IH_w = self.weight_IH[:, :size].reshape(self.weight_IH.data.size(0), size)
        IH_b = self.bias_IH[:, :size].reshape(self.bias_IH.data.size(0), size)
        HO_w = self.weight_HO[:size, :].reshape(size, self.weight_HO.data.size(1))
        H_state = torch.mm(input, IH_w) + IH_b
        H_state = torch.sigmoid(H_state)
        pred = torch.mm(H_state, HO_w)
        return pred

    def solve_output(self, feature, target):
        ftf = feature.T @ feature
        fty = feature.T @ target

        try:
            output_w = torch.linalg.lstsq(ftf, fty, driver = 'gelsd').solution.to(self.opts.device)
        except:
            output_w = torch.mm(torch.linalg.pinv(ftf), fty).to(self.opts.device)
            
        return output_w
    
    def data_loader(self, data):
        set_loader = torch_dataloader(mlp_dataset(data,self.output_dim,self.input_dim), cuda=self.usingCUDA )
        return set_loader
    

    def xfit(self, train_data, val_data):

        with torch.no_grad():
            min_vrmse = 9999
            loss = 9999
            train_loader = self.data_loader(train_data)
            val_loader = self.data_loader(val_data)
            
            train_x, train_y = None, None
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(torch.float32).to(self.opts.device)
                batch_y = batch_y.to(torch.float32).to(self.opts.device)
                train_x, train_y = batch_x.detach().clone(), batch_y.detach().clone()

            val_x, val_y = None, None
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(torch.float32).to(self.opts.device)
                batch_y = batch_y.to(torch.float32).to(self.opts.device)
                val_x, val_y = batch_x.detach().clone(), batch_y.detach().clone()
            


            fit_error = train_y.clone().detach()

            for i in trange(self.hidden_size):
                if i == self.hidden:
                    # success = True
                    Lambda = self.Lambdas[0]
                    new_hidden_weight = torch.empty(self.input_dim, 1).uniform_(-Lambda, Lambda).float().to(self.opts.device)
                    new_bias_weight = torch.empty(1, 1).uniform_(-Lambda, Lambda).float().to(self.opts.device)

                    new_fm = torch.mm(train_x, new_hidden_weight) + new_bias_weight
                    new_fm = torch.sigmoid(new_fm)

                    new_output_weight = self.solve_output(new_fm, fit_error)

                    if loss < self.tolerance:
                        break
                    else:
                        self.weight_IH = torch.cat((self.weight_IH, new_hidden_weight),1)
                        self.bias_IH = torch.cat((self.bias_IH, new_bias_weight), 1)
                        self.weight_HO = torch.cat((self.weight_HO, new_output_weight), 0)
                        self.hidden+=1
                
                if self.hidden == 1:
                    fm = torch.mm(train_x, self.weight_IH) + self.bias_IH
                    fm = torch.sigmoid(fm)
                    self.weight_HO = self.solve_output(fm, fit_error)

                pred = self.forward(train_x, i+1)
                fit_error = train_y - pred
                loss = self.loss_fn(pred, train_y).item()
                loss = np.sqrt(loss)

                vpred = self.forward(val_x, i+1)
                vloss = self.loss_fn(vpred, val_y).item()
                vloss = np.sqrt(vloss)

                if vloss < min_vrmse:
                    min_vrmse = vloss
                    self.best_hidden_size = self.hidden
                    self.best_HO = self.weight_HO.detach().clone()
                    self.logger.info('Found new best state')
                    self.logger.info('Best vmse: {:.4f}'.format(min_vrmse))

                self.logger.info('Hidden size: {} \t \nTraining RMSE: {:.8f} \t Validating RMSE: {:.8f} \t Best VRMSE: {:.8f} \t Best hidden size: {}'.format(
                    self.hidden, loss, vloss, min_vrmse, self.best_hidden_size))

                self.best_state = {
                        'hidden_size': self.weight_IH.data.size(1),
                        'best_hidden_size': self.best_hidden_size,
                        'best_HO': self.best_HO,
                        'weight_IH': self.weight_IH,
                        'bias_IH': self.bias_IH,
                        'weight_HO': self.weight_HO}
                
                self.fit_info.loss_list.append(loss)
                self.fit_info.vloss_list.append(vloss)
                self.fit_info.trmse = self.fit_info.loss_list[self.best_hidden_size - 1]
                self.fit_info.vrmse = min_vrmse
                
                gc.collect()

        return self.fit_info


    def predict(self, input, using_best = True):
        """
        Parameters
        ----------
        input : torch array-like shape, (n_samples, Input_dim)
            The data to be transformed.
            
        Returns
        -------
        output : numpy array-like shape, (n_samples, Output_dim).
        """
        if self.best_state is not None:
            self.load_state()

        with torch.no_grad():
            input = torch.tensor(input).float().to(self.opts.device)
            if using_best:
                pred= self.forward(input, self.best_hidden_size)
            else:
                pred = self.forward(input, self.hidden_size)
        return pred.cpu().numpy()

    def loader_pred(self, data_loader):
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred = self.forward(batch_x, self.best_hidden_size)
            x.append(batch_x.cpu())
            y.append(batch_y.cpu())
            pred.append(batch_pred.cpu())
        x = torch.cat(x, dim=0).detach().numpy()
        y = torch.cat(y, dim=0).detach().numpy()
        pred = torch.cat(pred, dim=0).detach().numpy()
        
        return x, y, pred

    def task_pred(self, task_data):
        data_loader = self.data_loader(task_data)
        x, y, pred = self.loader_pred(data_loader)
        
        return x, y, pred    