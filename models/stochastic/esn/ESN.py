import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from task.TaskLoader import Opt
from models.stochastic.Base import esnLayer
import gc
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np


class EchoStateNetwork(nn.Module):
    """Basic Leaky Integrator Echo State Network (ESN)

    'Jaeger, Herbert, et al. "Optimization and applications of echo state networks with leaky-integrator neurons." Neural networks 20.3 (2007): 335-352.'

    Neuron reservoir with recurrent connection and random
    weights. Leaky factor (or damping) ensure echoes in the network. No
    learning takes place in the reservoir, readout is left at the user's
    convience. The input processed by these ESN should be normalized in [-1, 1]

    Parameters
    ----------
    Leaky rate : float, optional
        Leaky (forget) factor for echoes, strong impact on the dynamic of the
        reservoir. Possible values between 0 and 1, default is 1 which is equal to basic ESN.

    weight_scaling : float, optional
        Spectral radius of the reservoir, i.e. maximum eigenvalue of the weight
        matrix, also strong impact on the dynamical properties of the reservoir.
        Classical regimes involve values around 1, default is 0.9

    discard_steps : int, optional
        Discard first steps of the timeserie, to allow initialization of the
        network dynamics.
    """

    def __init__(self, opts=None, logger=None):
        super(EchoStateNetwork, self).__init__()
        self.opts = opts
        self.logger = logger

        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.Discard_steps = opts.discard_steps

        self.Input_dim = 1 + opts.cov_dim
        self.Time_steps = opts.steps
        self.Output_dim = opts.H
        self.Hidden_Size = opts.hidden_size

        self.device = opts.device

        self.read_hidden = opts.read_hidden
        if self.read_hidden == 'all':
            self.readout_size = (
                self.Time_steps - self.Discard_steps) * self.Hidden_Size
        elif self.read_hidden == 'last':
            self.Discard_steps = 0
            self.readout_size = self.Hidden_Size
        else:
            raise ValueError(
                "Unknown read_hidden '{}'".format(self.read_hidden))

        self.fc_io = True if opts.fc_io == 'on' else False

        self.init_arch()
        self.loss_fn = nn.MSELoss()

        if self.opts.device == torch.device('cpu'):
            self.logger.info('Using CPU...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.opts.device)

        self.fit_info = Opt()

    def init_arch(self,):
        self.layer_esn = esnLayer(
            hidden_size=self.Hidden_Size,
            input_dim=self.Input_dim,
            nonlinearity=self.opts.nonlinearity,
            leaky_r=self.opts.leaky_r,
            weight_scale=self.opts.weight_scaling,
            iw_bound=self.opts.iw_bound,
            hw_bound=self.opts.hw_bound,
            device=self.device
        )
        self.layer_esn.freeze()

        if self.fc_io:
            self.readout_size += self.Input_dim * \
                (self.Time_steps - self.Discard_steps)

        self.readout = nn.Linear(
            self.readout_size, self.Output_dim)

    def io_check(self, hidden, x):
        if self.fc_io:
            _x = x[:, :, self.Discard_steps:]
            Hidden = torch.cat(
                (torch.flatten(_x, start_dim=1), hidden), dim=1)
            return Hidden
        else:
            return hidden

    def solve_output(self, Hidden_States, y):
        t_hs = torch.cat((torch.ones(Hidden_States.size(0), 1).to(
            self.device), Hidden_States), dim=1)
        HTH = t_hs.T @ t_hs
        HTY = t_hs.T @ y
        # ridge regression with pytorch
        I = (self.opts.lambda_reg * torch.eye(HTH.size(0))).to(self.device)
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

    def update_readout(self, H_state, x, y):
        _Hidden = self.io_check(H_state, x)
        W, tag = self.solve_output(_Hidden, y)
        self.readout.bias = nn.Parameter(W[:, 0], requires_grad=False)
        self.readout.weight = nn.Parameter(W[:, 1:], requires_grad=False)
        self.logger.info('Global LSM: {} \t L2 regular: {}'.format(
            tag, 'True' if self.opts.lambda_reg != 0 else 'False'))

    def forward(self, x):
        hidden = self.reservior_transform(x)
        hidden = self.io_check(hidden, x)
        output = self.readout(hidden)
        return output

    def reservior_transform(self, x):
        Hidden_States, last_state = self.layer_esn(x)
        if self.read_hidden == 'all':
            Hidden_States = torch.flatten(Hidden_States, start_dim=1)
        else:
            Hidden_States = last_state
        return Hidden_States

    def batch_transform(self, data_loader):
        h_states = []
        x = []
        y = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            # _h_states = self.reservior_transform(batch_x)
            h_states.append(self.reservior_transform(batch_x))
            x.append(batch_x)
            y.append(batch_y)
            
            torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
                    
        h_states = torch.cat(h_states, dim=0)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return h_states, x, y

    def xfit(self, train_loader, val_loader):
        # min_vmse = 9999
        h_states, x, y = self.batch_transform(train_loader)
        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()

        self.update_readout(h_states, x, y)
        pred = self.readout(self.io_check(h_states, x))
        self.fit_info.trmse = np.sqrt(self.loss_fn(pred, y).item())

        vh_states, val_x, val_y = self.batch_transform(val_loader)
        vpred = self.readout(self.io_check(vh_states, val_x))

        self.fit_info.vrmse = np.sqrt(self.loss_fn(vpred, val_y).item())

        self.xfit_logger()

        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()

        return self.fit_info

    def xfit_logger(self,):
        self.logger.info('Hidden size: {}'.format(self.Hidden_Size))
        self.logger.critical(
            'Training RMSE: {:.4g} \t Validating RMSE: {:.4g}'.format(self.fit_info.trmse, self.fit_info.vrmse))

    def predict(self, x):
        x = torch.tensor(x).to(torch.float32).to(self.device)
        output = self.forward(x)
        pred = output.detach().cpu().numpy()
        return pred
    
    def loader_pred(self, data_loader):
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred = self.forward(batch_x)
            x.append(batch_x)
            y.append(batch_y)
            pred.append(batch_pred)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()
        
        return x, y, pred