import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from task.util import savebest_checkpoint, load_checkpoint,plot_all_epoch,plot_xfit, os_makedirs

import numpy as np

from sklearn.metrics import mean_squared_error
import gc
from tqdm import trange
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.optim as optim
from task.TaskLoader import rnn_dataset, torch_dataloader
from task.TaskLoader import Opt
from tqdm import tqdm

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, opts=None, logger=None):
        super().__init__()
        self.opts = opts
        self.logger = logger

        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
            
        self.cell_type = opts.cell_type

        self.Lag_order = opts.lag_order
        self.Input_dim = 1 + opts.cov_dim
        self.Output_dim = opts.H
        self.Hidden_Size = opts.hidden_size
        self.num_layers = 1 if 'num_layers' not in opts.dict else opts.num_layers

        if self.cell_type == "rnn":
            self.Cell = nn.RNN(input_size=self.Input_dim, hidden_size=self.Hidden_Size,
                               num_layers=self.num_layers, dropout=0.0,
                               nonlinearity="relu", batch_first=True)
        if self.cell_type == "gru":
            self.Cell = nn.GRU(input_size=self.Input_dim, hidden_size=self.Hidden_Size,
                               num_layers=self.num_layers, batch_first=True)
        if self.cell_type == 'lstm':
            self.Cell = nn.LSTM(input_size=self.Input_dim, hidden_size=self.Hidden_Size,
                                num_layers=self.num_layers, dropout=0.0,
                                batch_first=True)

        self.fc = nn.Linear(self.Hidden_Size, self.Output_dim)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=opts.learning_rate)
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, opts.step_lr, gamma=0.9)
        self.loss_fn = nn.MSELoss()

        self.epochs = self.opts.epochs

        # self.opts.plot_dir = os.path.join(opts.model_fit_dir, 'figures')
        # # create missing directories
        # os_makedirs(self.opts.plot_dir)

        if self.opts.device == torch.device('cpu'):
            self.logger.info('Using CPU...')
        else:
            self.logger.info('Using Cuda...')
            self.usingCUDA = False
            self.to(self.opts.device)
            self.usingCUDA = True
            
        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        self.best_train_idx = -1
        
        self.opts.restore = False if 'restore' not in opts.dict else True
        
    def initHidden(self, input):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            batchSize = int(input.batch_sizes[0])
        else:
            batchSize = input.data.size(0)
        result = torch.empty(1,
                            batchSize, self.Hidden_Size).float().to(self.opts.device)
        result = nn.init.normal_(result, std=0.015)
        return result

    def forward(self, input):
        # input: shape[batch,input_dim,time_step] need to be [batch,time_step,input_dim]
        # h_state: shape[layer_num*direction,batch,hidden_size]
        # rnn_output: shape[batch,time_sequence_length,hidden_size]
        
        input = torch.permute(input, (0,2,1))
        
        if self.cell_type == 'lstm':
            h_state = self.initHidden(input)
            c_state = self.initHidden(input)
            RNN_Output, (h_state_n, c_state_n) = self.Cell(
                input, (h_state, c_state))
        elif self.cell_type == 'rnn' or 'gru':
            h_state = self.initHidden(input)
            RNN_Output, h_state_n = self.Cell(input, h_state)
        Outputs = self.fc(RNN_Output)
        # Outputs : shape[batch,time_sequence_length,output_dim]
        return Outputs

    def data_loader(self, data, _batch_size = None):
        '''
        Transform the numpy array data into the pytorch data_loader
        '''
        data_batch_size = self.opts.batch_size if _batch_size is None else _batch_size
        set_loader = torch_dataloader(rnn_dataset(data, self.Output_dim, self.Lag_order,self.Input_dim), batch_size= data_batch_size,cuda= self.usingCUDA)
        return set_loader
    


    def xfit(self,train_data, val_data, restore_file=None):
        # update self.opts
        if restore_file is not None and os.path.exists(restore_file) and self.opts.restore:
            self.logger.info(
                'Restoring parameters from {}'.format(restore_file))
            load_checkpoint(restore_file, self, self.optimizer)

        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        
        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
        
        min_vmse = 9999
        train_len = len(train_loader)
        loss_summary = np.zeros((train_len * self.epochs))
        loss_avg = np.zeros((self.epochs))
        vloss_avg = np.zeros_like(loss_avg)

        epoch = 0
        for epoch in trange(self.epochs):
            self.logger.info(
                'Epoch {}/{}'.format(epoch + 1, self.epochs))
            mse_train = 0
            loss_epoch = np.zeros(train_len)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(torch.float32).to(self.opts.device)
                batch_y = batch_y.to(torch.float32).to(self.opts.device) # shape[batch,output_dim,time_sequence_length]
                self.optimizer.zero_grad()
                y_pred = self(batch_x) # shape[batch,time_sequence_length,output_dim]
                y_pred = torch.flatten(y_pred, end_dim=1)  # shape[batch*time_sequence_length,output_dim]
                batch_y = torch.flatten(torch.permute(batch_y, (0,2,1)), end_dim=1)
                # y_pred = y_pred.squeeze(1)
                loss = self.loss_fn(y_pred, batch_y)
                loss.backward()
                mse_train += loss.item()
                loss_epoch[i] = loss.item()
                self.optimizer.step()
            
            mse_train = mse_train / train_len
            loss_summary[epoch * train_len:(epoch + 1) * train_len] = loss_epoch
            loss_avg[epoch] = mse_train
            self.fit_info.loss_list.append(mse_train)

            self.epoch_scheduler.step()
            
            with torch.no_grad():
                mse_val = 0
                preds = []
                true = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(torch.float32).to(self.opts.device)
                    batch_y = batch_y.to(torch.float32).to(self.opts.device)
                    output = self(batch_x)
                    output = output[:,-1, :]
                    batch_y = batch_y[:,:,-1]
                    # output = output.squeeze(1)
                    preds.append(output.detach().cpu().numpy())
                    true.append(batch_y.detach().cpu().numpy())
                    mse_val += self.loss_fn(output,
                                         batch_y).item()
                mse_val = mse_val / len(val_loader)
            vloss_avg[epoch] = mse_val
            self.fit_info.vloss_list.append(mse_val)
            
            preds = np.concatenate(preds)
            true = np.concatenate(true)

            self.logger.info('Current training loss: {:.4E} \t validating loss: {:.4E}'.format(mse_train,mse_val))
            
            vmse = mean_squared_error(true, preds)
            self.logger.info('Current vmse: {:.4E}'.format(vmse))
            if vmse < min_vmse:
                min_vmse = vmse
                self.best_train_idx = epoch
                self.logger.info('Found new best state')
                self.logger.info('Best vmse: {:.4E}'.format(min_vmse))
                if self.opts.restore:
                    savebest_checkpoint({
                        'epoch': epoch,
                        'cv': self.opts.cv,
                        'state_dict': self.state_dict(),
                        'optim_dict': self.optimizer.state_dict()}, checkpoint=self.opts.model_fit_dir)
                    self.logger.info(
                        'Checkpoint saved to {}'.format(self.opts.model_fit_dir))                        
                else:
                    self.best_state = self.state_dict()

        # plot_all_epoch(loss_summary[:(
        #     epoch + 1) * train_len], self.opts.dataset + '_loss', self.opts.plot_dir)
        self.fit_info.tloss = self.fit_info.loss_list[self.best_train_idx]
        self.fit_info.vloss = self.fit_info.vloss_list[self.best_train_idx]
        
        return self.fit_info
        

    def predict(self, x, using_best=True):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''
        # test_batch: shape: [full-len, sample, dim]
        best_pth = os.path.join(self.opts.model_fit_dir, 'best.cv{}.pth.tar'.format(self.opts.cv))
        if os.path.exists(best_pth) and using_best:
            self.logger.info('Restoring best parameters from {}'.format(best_pth))
            load_checkpoint(best_pth, self, self.optimizer)
        elif using_best:
            self.load_state_dict(self.best_state)

        x = torch.tensor(x).to(torch.float32).to(self.opts.device)
        output = self(x)
        output = output[:,:, -1]
        # output = output.squeeze(1)
        pred = output.detach().cpu().numpy()

        return pred
    
    def loader_pred(self, data_loader):
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.opts.device)
            batch_y = batch_y[:,:,-1]
            batch_pred = self.forward(batch_x)
            batch_pred = batch_pred[:,-1,:]
            x.append(batch_x.cpu())
            y.append(batch_y.cpu())
            pred.append(batch_pred.cpu())
        x = torch.cat(x, dim=0).detach().numpy()
        y = torch.cat(y, dim=0).detach().numpy()
        pred = torch.cat(pred, dim=0).detach().numpy()
        
        return x, y, pred
    
    def task_pred(self, task_data):
        self.eval()
        data_loader = self.data_loader(task_data)
        x, y, pred = self.loader_pred(data_loader)
        
        return x, y, pred    