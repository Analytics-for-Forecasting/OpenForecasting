import os

from numpy.lib.function_base import select

# from torch.utils import data

# from task.util import os_rmdirs, set_logger

from task.dataset import mlp_dataset, dnn_dataset
from task.dataset import difference, create_dataset
from task.dataset import deepAR_dataset, deepAR_weight, deepAR_WeightedSampler

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import numpy as np

# from models.Config import model_opts_dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from collections.abc import Mapping
from task.dataset import de_scale

# from task.metric import rmse

class Opt(object):
    def __init__(self, init = None):
        super().__init__()
        
        if init is not None:
            self.merge(init)
        
    def merge(self, opts, ele_s = None):
        new = vars(opts)
        if ele_s is None:
            for key in new:
                if not key in self.dict:
                    self.dict[key] = new[key]
        else:
            for key in ele_s:
                if not key in self.dict:
                    self.dict[key] = new[key]
                
    def update(self, opts):
        if isinstance(opts, Mapping):
            new = opts
        else:
            assert isinstance(opts, object)
            new = vars(opts)
        for key in new:
            if not key in self.dict:
                raise ValueError(
                "Unknown config key '{}'".format(key))
            self.dict[key] = new[key]
            
    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__

class Params(Opt):
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, args = None,):
        if args is not None:
            self.merge(args)        
        # if json_path is not None:
        #     with open(json_path) as f:
        #         params = json.load(f)
        #         assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
        #         self.__dict__.update(params)
    # def update(self, dict):
    #     for key in dict:
    #         if not key in self.dict:
    #             raise ValueError(
    #             "Unknown config key '{}'".format(key))
    #         self.dict[key] = dict[key]

class TaskDataset(Opt):
    def __init__(self, args):
        super().__init__()
        self.info = Opt()
        self.info.normal = args.normal
        self.info_config()
        self.info.H = args.H
        self.args = args
    
    def info_config(self, ):
        pass
    
    def pack_dataset(self, raw_ts):
        
        sub = Opt()
       
        _dataset, train_idx, valid_idx, test_idx = self.data_split(raw_ts)
        
        sub.scaler = Scaler(self.info.normal)
        sub.scaler.fit(raw_ts)
        
        #for testing
        sub.orig = _dataset
        
        dataset = sub.scaler.transform(_dataset)
                
        train_loader, valid_loader, test_loader = self.data_loader(dataset[train_idx], dataset[valid_idx], dataset[test_idx])
                
        sub.train_loader = train_loader
        sub.valid_loader = valid_loader
        sub.test_loader = test_loader
        sub.steps = self.info.steps
        sub.H = self.info.H
        
        # sub.test_input = test_input
        # sub.test_target = test_target
        # to do check the torch.cat([batch_x, for b_x in test_loader]) is test_input
        # to do replace, model.predict with model.loader_pre    
        
        # generate the results of naive forecasting method.
        last_naive, avg_naive = self.get_naive_pred(sub)
        sub.lastNaive = last_naive
        sub.avgNaiveError = avg_naive
    
        return sub
    
    def get_naive_pred(self, sub):
        tx = []
        for batch_x, _ in sub.test_loader:
            tx.append(batch_x)
        tx = torch.cat(tx, dim=0).detach().cpu().numpy()
        if tx.ndim == 3:
            tx = tx[:, 0, :]
        
        _tx = de_scale(sub, tx, tag='input')
        
        last_naive = _tx[:, -1]
        
        avg_naive = np.zeros_like(last_naive)
        lag_order = _tx.shape[1]
        for t in range(1, lag_order):
            step_error = _tx[:,t] - _tx[:, t-1]
            step_error = np.abs(step_error)
            avg_naive += step_error
        
        avg_naive = avg_naive / (lag_order - 1)
        
        return last_naive, avg_naive
        
    def data_split(self,raw_ts):
        dataset = create_dataset(raw_ts, self.info.steps + self.info.H - 1)
        
        tscv = TimeSeriesSplit(n_splits=self.args.k-1)
        *lst, last = tscv.split(dataset)
        train_idx, test_idx = last
        _train = dataset[train_idx]
        train_tscv = TimeSeriesSplit(n_splits=self.args.k-1)
        *lst, last = train_tscv.split(_train)
        train_idx, valid_idx = last
        
        return dataset, train_idx, valid_idx, test_idx

    def data_loader(self, train_data, valid_data, test_data):
        train_loader, valid_loader,test_loader = None, None, None
        dataform = None

        arch = self.arch

        if arch == 'deepar':
            self.args.train_window = self.info.steps+self.info.H
            self.args.test_window = self.args.train_window
            self.args.predict_start = self.info.steps
            self.args.predict_steps = self.info.H

            train_set, train_input, _ = deepAR_dataset(
                train_data, train=True, h=self.info.H, steps=self.info.steps, sample_dense=self.args.sample_dense)
            valid_set, _, _ = deepAR_dataset(
                valid_data, train=True, h=self.info.H, steps=self.info.steps, sample_dense=self.args.sample_dense)
            _, test_input, test_target = deepAR_dataset(
                test_data, train=False, h=self.info.H, steps=self.info.steps, sample_dense=self.args.sample_dense)
            test_target = test_target[:, self.args.predict_start:]

            _, train_v = deepAR_weight(train_input, self.info.steps)
            train_sample = deepAR_WeightedSampler(train_v)

            self.info.batch_size = 128
            train_loader = DataLoader(
                train_set, batch_size=self.info.batch_size, sampler=train_sample, pin_memory=self.args.cuda, num_workers=4)
            valid_loader = DataLoader(valid_set, batch_size=valid_set.samples,
                                    sampler=RandomSampler(valid_set), pin_memory=self.args.cuda, num_workers=4)
            # to do: test_loader
        else:
            if arch == 'rnn' or arch == 'cnn':
                dataform = dnn_dataset
            elif arch == 'mlp' or 'statistic':
                dataform = mlp_dataset
            else:
                assert False
                
            train_set, _, _ = dataform(train_data, self.info.H, self.info.steps)
            valid_set, _, _ = dataform(valid_data, self.info.H, self.info.steps)
            test_set, _, _ = dataform(
                test_data, self.info.H, self.info.steps)

            
            if 'batch_size' in self.info.dict:
                train_batch_size = self.info.batch_size if self.info.batch_size < train_set.samples else train_set.samples
                valid_batch_size = self.info.batch_size if self.info.batch_size < valid_set.samples else valid_set.samples
                test_batch_size = self.info.batch_size if self.info.batch_size < test_set.samples else test_set.samples
            else:
                train_batch_size = train_set.samples
                valid_batch_size = valid_set.samples
                test_batch_size = test_set.samples
        
            train_loader = DataLoader(train_set, batch_size= train_batch_size , pin_memory=self.args.cuda, shuffle=False, sampler=SequentialSampler(train_set))
            
            valid_loader = DataLoader(valid_set, batch_size=valid_batch_size, pin_memory=self.args.cuda, shuffle=False, sampler=SequentialSampler(valid_set))

            test_loader = DataLoader(test_set, batch_size=test_batch_size, pin_memory=self.args.cuda, shuffle=False, sampler=SequentialSampler(test_set))

        return train_loader, valid_loader, test_loader


def get_dataset(params):
    rawdataset_path = './data/{}/{}.H{}.npy'.format(
        params.datafolder, params.dataset, params.H)
    if params.diff:
        dataset_path = './data/{}/{}.H{}.diff.npy'.format(
            params.datafolder, params.dataset, params.H)
    else:
        dataset_path = rawdataset_path

    if os.path.exists(rawdataset_path) and os.path.exists(dataset_path):
        raw_dataset = np.load(rawdataset_path)
        dataset = np.load(dataset_path) if params.diff else raw_dataset
    else:
        raw_ts = np.load(
            './data/{}/{}.npy'.format(params.datafolder, params.dataset))
        raw_ts = raw_ts.reshape(-1)
        df_ts = difference(raw_ts)

        raw_dataset = create_dataset(raw_ts, look_back=params.steps + params.H - 1)
        if params.diff:
            dataset = create_dataset(
                df_ts, look_back=params.steps + params.H - 1)
        else:
            dataset = raw_dataset
    assert raw_dataset.shape == dataset.shape

    # if params.datafolder == 'elect.price':
    #     test_section = 1440 + 168
    #     test_ts = raw_ts[-test_section:]
    #     val_ts = raw_ts[-2*1440 - 168:-1440]
        
    #     test = create_dataset(test_ts,  look_back=params.steps + params.H - 1)
    #     # print(test[0,:])
    #     val = create_dataset(val_ts,  look_back=params.steps + params.H - 1)
    #     # print(val[0,:])
        
    #     N = raw_dataset.shape[0]
    #     T = test.shape[0]
    #     V = val.shape[0]
        
    #     test_idx = np.array(range(N-T,N))
    #     valid_idx = np.array(range(N-T-V,N-T))
    #     train_idx = np.array(range(N-T-V))
    # else:
    tscv = TimeSeriesSplit(n_splits=params.k-1)
    *lst, last = tscv.split(dataset)
    train_idx, test_idx = last
    _train = dataset[train_idx]
    train_tscv = TimeSeriesSplit(n_splits=params.k-1)
    *lst, last = train_tscv.split(_train)
    train_idx, valid_idx = last

    return raw_ts, raw_dataset, dataset, train_idx, valid_idx, test_idx

def merge_dataset_info(params):

    # dataset_params_path = 'data/{}/lag_settings.json'.format(params.datafolder)
    # params.load_json(dataset_params_path)

    # dataset_info = params.datasets[params.dataset]

    # params.steps = dataset_info['lag_order']
    # # params.normal = dataset_info['normal']
    # params.cov_dim = dataset_info['cov_dim']
    params.period = 1 if 'period' not in params.dict else params.period

    return params



def transform_dataset(params):
    
    params = merge_dataset_info(params)
    
    raw_ts, raw_dataset, dataset, train_idx, valid_idx, test_idx = get_dataset(params)
    # raw_test_data = raw_dataset[test_idx]
    params.scaler.fit(raw_ts)
    dataset = params.scaler.transform(dataset)
    train_data, valid_data, test_data = dataset[train_idx], dataset[valid_idx], dataset[test_idx]
    
    params, train_loader, valid_loader, test_input, test_target = data_loader(params, train_data, valid_data, test_data)

    return params, train_loader, valid_loader, test_input, test_target


def data_loader(args, train_data, valid_data, test_data):
    train_loader, valid_loader, test_input, test_target = None, None, None, None
    dataform = None

    arch = args.model_arch

    if arch == 'deepar':
        args.train_window = args.steps+args.H
        args.test_window = args.train_window
        args.predict_start = args.steps
        args.predict_steps = args.H

        train_set, train_input, _ = deepAR_dataset(
            train_data, train=True, h=args.H, steps=args.steps, sample_dense=args.sample_dense)
        valid_set, _, _ = deepAR_dataset(
            valid_data, train=True, h=args.H, steps=args.steps, sample_dense=args.sample_dense)
        _, test_input, test_target = deepAR_dataset(
            test_data, train=False, h=args.H, steps=args.steps, sample_dense=args.sample_dense)
        test_target = test_target[:, args.predict_start:]

        _, train_v = deepAR_weight(train_input, args.steps)
        train_sample = deepAR_WeightedSampler(train_v)

        args.batch_size = 128
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, sampler=train_sample, pin_memory=args.cuda, num_workers=4)
        valid_loader = DataLoader(valid_set, batch_size=valid_set.samples,
                                  sampler=RandomSampler(valid_set), pin_memory=args.cuda, num_workers=4)
    else:
        if arch == 'rnn' or arch == 'cnn':
            dataform = dnn_dataset
        elif arch == 'mlp':
            dataform = mlp_dataset
        else:
            assert False

        train_set, _, _ = dataform(train_data, args.H, args.steps)
        valid_set, _, _ = dataform(valid_data, args.H, args.steps)
        test_set, _, _ = dataform(
            test_data, args.H, args.steps)

        
        if 'batch_size' in args.dict:
            train_batch_size = args.batch_size if args.batch_size < train_set.samples else train_set.samples
            valid_batch_size = args.batch_size if args.batch_size < valid_set.samples else valid_set.samples
        else:
            train_batch_size = train_set.samples
            valid_batch_size = valid_set.samples
    
        
        train_loader = DataLoader(train_set, batch_size= train_batch_size , pin_memory=args.cuda, shuffle=False, sampler=SequentialSampler(train_set))
        
        valid_loader = DataLoader(valid_set, batch_size=valid_batch_size, pin_memory=args.cuda, shuffle=False, sampler=SequentialSampler(valid_set))

    return args, train_loader, valid_loader, test_input, test_target

    
class Scaler:
    def __init__(self, normal):
        if normal:
            self.scaler = StandardScaler()
        else:
            self.scaler= MinMaxScaler(feature_range=(-1, 1))
            
    def fit(self, dataset):
        # check 1d
        assert len(dataset.shape) == 1
        data = dataset.reshape(-1,1)
        self.scaler.fit(data)
        
    def transform(self, dataset):
        # check 2d
        assert len(dataset.shape)  == 2
        
        temp = np.empty_like(dataset)
        
        for c in range(dataset.shape[1]):
            column = dataset[:,c].reshape(-1,1)
            column = self.scaler.transform(column).reshape(-1,)
            temp[:,c]= column
        
        return temp
            
    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
        
    
    def inverse_transform(self, dataset):
        assert len(dataset.shape)  == 2
        
        temp = np.empty_like(dataset)
        
        for c in range(dataset.shape[1]):
            column = dataset[:,c].reshape(-1,1)
            column = self.scaler.inverse_transform(column).reshape(-1,)
            temp[:,c]= column
        
        return temp
    
