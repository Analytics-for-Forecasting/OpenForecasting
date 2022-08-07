"""
ARIMA with auto model selection
"""
import os
import sys

# from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np
from task.dataset import unpadding, create_dataset
import pmdarima as pm

from task.TaskLoader import Opt

from tqdm import trange,tqdm
import torch


class ARIMA():
    def __init__(self, params=None, logger=None):
        self.params = params
        self.logger = logger
        
        self.max_length = self.params.period * 2
        
        self.refit = params.refit
        
        self.device = torch.device('cpu')
        
        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
    
    def check_history(self,):
        if self.history.shape[0] >  self.max_length:
            self.history = self.history[-self.max_length:]
        
        if self.history.min() <= 0:
            self.constant = self.history.min() * -1.05 
        else:
            self.constant = 0
        
        self.history = self.history + self.constant
        
        if self.history.min() < 0:
            raise ValueError('Current history min() is: {}'.format(self.history.min()))
                    
        return self.history, self.constant
            
        
    def xfit(self, train_loader, val_loader):
        batch_x, batch_y = None, None
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.numpy()
            batch_y = batch_y.numpy()
            # print(batch_x.shape)
            # print(batch_y.shape)

        data_train = np.concatenate((batch_x,batch_y), axis=1)
        # print(data_train[-1, :3])
        # print(data_train[-1, -3:])
        # print(data_train.shape)

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.numpy()
            batch_y = batch_y.numpy()
            # print(batch_x.shape)
            # print(batch_y.shape)
            
        data_valid = np.concatenate((batch_x,batch_y), axis=1)
        # print(data_valid[0, :3])
        # print(data_valid[0, -3:])

        # print(data_valid[-1, :3])
        # print(data_valid[-1, -3:])
        # print(data_valid.shape)
        data_fit = np.concatenate((data_train, data_valid), axis=0)
        
        self.data_fit = unpadding(data_fit)
        # print(data_fit[-3:])
        # self.model = pm.auto_arima(data_fit[:( self.data_fit.shape[0] - (self.params.H - 1) )], seasonal=False)
        
        self.history = self.data_fit[:( self.data_fit.shape[0] - (self.params.H - 1) )]
                
        self.check_history()
        
        return Opt()
    
    def gen_model(self,):
        self.model = pm.auto_arima(self.history, seasonal=False,  error_action="ignore")
    
    def _predict(self, H):
        pred = self.model.predict(H)
        return pred
    
    def predict(self, input):
        H = self.params.H
        
        test_samples = input.shape[0]
        pred_sec = H + input.shape[0] - 1
        
        # print(history[-3:])
        
        if self.refit == False:
            self.gen_model()
            yPred = self._predict(pred_sec) - self.constant
            yPred = create_dataset(yPred, H - 1)
        else:
            yPred = np.empty((test_samples, H))
            for i in trange(test_samples):
                self.gen_model()
                yPred[i,:]= self._predict(H) - self.constant
                self.history = np.concatenate((self.history[1:], input[i,-1].reshape(-1,)), axis=0)
                self.history, self.constant = self.check_history()
                # print(history[-3:])
        return yPred

    def loader_pred(self, data_loader):
        x = []
        y = []
        # pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            # batch_pred = self.predict(batch_x)
            x.append(batch_x)
            y.append(batch_y)
            # pred.append(batch_pred)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred = self.predict(x)
        
        return x, y, pred    