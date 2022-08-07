import os
import sys

from numpy.lib.function_base import select
from ray.tune import logger
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import shutil
from tqdm import trange
import statistics
import numpy as np
import torch
from task.TaskLoader import Opt
from task.TaskTuner import StocHyperTuner as HyperTuner
from task.dataset import de_scale
from task.util import os_makedirs, os_rmdirs, set_logger
import importlib
from tqdm.std import tqdm
# from task.metric import rmse, mape, smape
from task.util import plot_xfit

import math

from shutil import copyfile

def copy_check(src_file, dst_file):
    if os.path.exists(src_file):
        print('Find: {}'.format(src_file))
        copyfile(src_file, dst_file)
        if os.path.exists(dst_file):
            print("Copying to {}".format(dst_file))
        else:
            raise ValueError('Fail in copying {}'.format(src_file))

class Task(Opt):
    def __init__(self, args):
        # self.opts = Opt()
        # self.opts.merge(args)

        self.exp_config_path = importlib.import_module('data.{}.config_{}'.format(
            args.datafolder.replace('/', '.'), args.dataset))  # 引入配置
        self.data_config(args)
        self.model_config(args)
        self.exp_config(args)
        self.data_subconfig()

    def data_config(self, args):
        self.data_name = args.dataset
        data_opts = getattr(self.exp_config_path, args.dataset + '_data')
        self.data_opts = data_opts(args)

    def data_subconfig(self,):
        self.data_opts.arch = self.model_opts.arch
        self.data_opts.sub_config()

    def model_config(self, args):
        self.model_name = args.model
        model_opts = getattr(self.exp_config_path,
                             args.model)       # getattr 返回类的属性值
        self.model_opts = model_opts()
        self.model_opts.hyper.merge(opts=self.data_opts.info)
        self.model_opts.hyper.H = args.H

        if self.model_opts.arch == 'cnn':
            if not self.model_name == 'clstm':
                self.model_opts.hyper.kernel_size = math.ceil(self.model_opts.hyper.steps / 4)
    
    def model_import(self,):
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        # model = model(self.model_opts.hyper, self.logger)  # transfer model in the conduct func.
        return model

    def exp_config(self, args):
        cuda_exist = torch.cuda.is_available()
        if cuda_exist and args.cuda:
            self.device = torch.device('cuda:{}'.format(args.gid))
        else:
            self.device = torch.device('cpu')

        if 'statistic' in vars(self.model_opts):
            self.device = torch.device('cpu')

        self.exp_dir = 'trial' if args.test == False else 'test'

        if args.mo is not None:
            self.exp_dir = os.path.join(self.exp_dir, args.mo)

        self.exp_dir = os.path.join(
            self.exp_dir, 'normal') if self.data_opts.info.normal else os.path.join(self.exp_dir, 'minmax')

        assert args.diff == False  # Not support differential preprocess yet!
        # self.exp_dir += '_diff'

        self.exp_dir = os.path.join(self.exp_dir, args.dataset)

        task_name = os.path.join('{}.refit'.format(args.model), 'h{}'.format(
            args.H)) if 'refit' in self.model_opts.hyper.dict and self.model_opts.hyper.refit else os.path.join('{}'.format(args.model), 'h{}'.format(args.H))

        self.task_dir = os.path.join(self.exp_dir, task_name)

        if args.test and args.logger_level != 20:
            self.logger_level = 50  # equal to critical
        else:
            self.logger_level = 20  # equal to info

        self.rep_times = args.rep_times

        if args.test and args.clean:
            os_rmdirs(self.task_dir)
        os_makedirs(self.task_dir)

        self.measure_dir = os.path.join(self.task_dir, 'eval_results')
        os_makedirs(self.measure_dir)



        self.model_opts.hyper.device = self.device
        self.tune = args.tune

    def logger_config(self, dir,stage, cv, sub_count):
        log_path = os.path.join(dir, 'logs',
                                '{}.cv{}.series{}.log'.format(stage,cv, sub_count))
        log_name = '{}.series{}.cv{}.{}'.format(
            self.data_name, sub_count, cv, self.model_name)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger
    
    def tuning(self,):
        try:
            self.tune = True
            for sub_count, series_Pack in enumerate(tqdm(self.data_opts.seriesPack)):
                assert sub_count == series_Pack.index
                self.tuner_dir = os.path.join(self.task_dir, 'tuner')
                os_makedirs(self.tuner_dir)
                tuner_path = os.path.join(
                    self.tuner_dir, 'series{}.best.pt'.format(sub_count))

                self.model_opts.tuner.dir = self.tuner_dir

                # logger = self.logger_config(self.tuner_dir,'tuning','T',sub_count)
                if not os.path.exists(tuner_path):
                    series_tuner = HyperTuner(self.model_opts, logger, series_Pack)
                    series_tuner.conduct()
                    best_hyper = series_tuner.best_config
                    logger.critical('-'*80)
                    logger.critical('Tuning complete.')
                    torch.save(best_hyper, tuner_path)
                else:
                    src_path = tuner_path
                    series_dir = os.path.join(self.task_dir, 'series{}'.format(sub_count))
                    os_makedirs(series_dir)
                    dst_path = os.path.join(series_dir, 'series{}.best.pt'.format(sub_count))
                    copyfile(src_path, dst_path)
                    
                    # best_hyper = torch.load(tuner_path)

                # best_hyper is a dict type
                # for (arg, value) in best_hyper.items():
                #     logger.info("Tuning Results:\t %s - %r", arg, value)
        except:
            raise ValueError('{}\nGot an error on tuning.\n{}'.format('!'*50,'!'*50))
                                
    def conduct(self,):
        # init and mkdir taskdir
        # generate the subPack dataset
        if 'innerTuning' in self.model_opts.dict:
            if self.model_opts.innerTuning == True:
                self.tune = False
            
        if self.tune:
            self.tuning()
            
        for sub_count, series_Pack in enumerate(tqdm(self.data_opts.seriesPack)):
            # self.task_dir = os.path.join(self.task_dir, 'series_{}'.format(sub_count))
            series_dir = os.path.join(self.task_dir, 'series{}'.format(sub_count))
            os_makedirs(series_dir)
            s_m_dir = os.path.join(series_dir, 'eval_results')
            os_makedirs(s_m_dir)
            
            
            for i in trange(self.rep_times):
                assert sub_count == series_Pack.index

                result_file = os.path.join(
                    self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))

                if os.path.exists(result_file):
                    
                    dst_file = os.path.join(s_m_dir, 'results_{}.series_{}.npy'.format(i, sub_count))
                    
                    # copyfile(result_file, dst_file)
                    
                    continue
                if i > 0 and 'statistic' in self.model_opts.dict:
                    assert self.model_opts.statistic
                    result0 = str(os.path.join(
                        self.measure_dir, 'results_{}.series_{}.npy'.format(0, sub_count)))
                    shutil.copy(result0, result_file)
                    continue

                # loading the best paramters:
                self.logger = self.logger_config(self.task_dir,'train',i,sub_count)
                self.logger.critical('*'*80)
                self.logger.critical('Dataset: {}\t Model:{} \t H: {}\t Trail: {}'.format(
                    self.data_name, self.model_name, self.model_opts.hyper.H, i))
                    
                self.conduct_iter(i, series_Pack, result_file)

        
    def conduct_iter(self, i, subPack, result_file):
        try:
            if self.tune:
                tuner_path = os.path.join(
                    self.tuner_dir, 'series{}.best.pt'.format(subPack.index))
                best_hyper = torch.load(tuner_path)
                if not os.path.exists(tuner_path):
                    raise ValueError(
                        'Invalid tuner path: {}'.format(tuner_path))
                self.model_opts.hyper.update(best_hyper)
                self.logger.info("Updating tuning result complete.")
                self.logger.critical('-'*80)
                
            self.seed = i
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            self.logger.critical(
                'For {}th-batch-trainingLoading, loading the sub-datasets {}'.format(i, subPack.index))
            self.logger.critical('-'*80)
            
            # Attention! sub.H can be different among multiple series in some cases.
            self.model_opts.hyper.H = subPack.H
            self.model_opts.hyper.task_dir = self.task_dir
            model = self.model_import()
            model = model(self.model_opts.hyper, self.logger) 

            self.logger.critical('Loading complete.')
            self.logger.critical(f'Model: \n{str(model)}')
            
            fit_info = model.xfit(subPack.train_loader, subPack.valid_loader)
            self.plot_fitInfo(fit_info, subId=subPack.index, cvId= i)
                
            _, tgt, pred = model.loader_pred(subPack.test_loader)

            _tgt, _pred = de_scale(
                subPack, tgt), de_scale(subPack, pred)

            # _dataset = subPack.orig
            # _orig_tgt = _dataset[-_tgt.shape[0]:, -subPack.H:]
            # diff = rmse(_orig_tgt, _tgt)
            # print(diff)
            # to do, including differential dataset.

            self.logger.critical('-'*50)

            np.save(result_file,
                    (_tgt, _pred))
        except:
            self.logger.exception('{}\nGot an error on conduction.\n{}'.format('!'*50,'!'*50))
            raise SystemExit()

    def plot_fitInfo(self, fit_info, subId, cvId):
        if 'loss_list' in fit_info.dict and 'vloss_list' in fit_info.dict:
            plot_dir = os.path.join(self.task_dir, 'figures')
            
            # plot_dir = self.plot_dir if len(self.data_opts.seriesPack) == 1 else os.path.join(self.plot_dir, 'series{}'.format(subId))
            
            os_makedirs(plot_dir)
            
            plot_xfit(fit_info,                  'cv{}.series{}'.format(cvId,subId), plot_dir)
            self.logger.critical('Ploting complete. Saving in {}'.format(plot_dir))
        
        
    def evaluation(self, metrics=['mase']):
        # try:
        self.metrics = metrics
        eval_list = []
        for sub_count in range(self.data_opts.info.num_series):
            series_dir = os.path.join(self.task_dir, 'series{}'.format(sub_count))
            os_makedirs(series_dir)
            
            s_m_dir = os.path.join(series_dir, 'eval_results')
            os_makedirs(s_m_dir)
            
            s_l_dir = os.path.join(series_dir, 'logs')
            os_makedirs(s_l_dir)
            
            s_f_dir = os.path.join(series_dir, 'figures')
            os_makedirs(s_f_dir)
            
            s_t_dir = os.path.join(series_dir, 'tuner')
            os_makedirs(s_t_dir)
            
            
            file_name = 'series{}.best.pt'.format(sub_count)
            dst_file = os.path.join(s_t_dir,file_name)
            src_file = os.path.join(self.task_dir, 'tuner', file_name)
            copy_check(src_file,dst_file)
                    
            
            for i in range(self.rep_times):

                src_file = os.path.join(
                    self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))
                
                dst_file = os.path.join(s_m_dir, 'results_{}.series_{}.npy'.format(i, sub_count))
                
                copy_check(src_file,dst_file)
                
                file_name = 'train.cv{}.series{}.log'.format(i,sub_count)
                dst_file = os.path.join(s_l_dir,file_name)
                src_file = os.path.join(self.task_dir, 'logs', file_name)
                copy_check(src_file,dst_file)
                
                file_name = 'eval.cv{}.series{}.log'.format(i,sub_count)
                dst_file = os.path.join(s_l_dir,file_name)
                src_file = os.path.join(self.task_dir, 'logs', file_name)
                copy_check(src_file, dst_file)


                file_name = 'cv{}.series{}.xfit.png'.format(i,sub_count)
                dst_file = os.path.join(s_f_dir,file_name)
                src_file = os.path.join(self.task_dir, 'figures', file_name)
                copy_check(src_file, dst_file)
                    
                file_name = 'cv{}.series{}.loss.npy'.format(i,sub_count)
                dst_file = os.path.join(s_f_dir,file_name)
                src_file = os.path.join(self.task_dir, 'figures', file_name)
                copy_check(src_file, dst_file)
                    

        # print('@'*80)
        # print('Dataset:\t{}\tModel:\t{}\tH:\t{}\tTrail:\t{}'.format(
        #     self.data_name, self.model_name, self.data_opts.info.H, self.rep_times))
        # eval_data = np.concatenate(eval_list, axis=0)
        # eval_return = []
        # for i, eval_name in enumerate(self.metrics):
        #     i_data = eval_data[:, i].tolist()

        #     if self.rep_times * self.data_opts.info.num_series > 1:
        #         mean = statistics.mean(i_data)
        #         std = statistics.stdev(i_data, mean)
        #     else:
        #         mean = i_data[0]
        #         std = 0

        #     print('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
        #         eval_name, mean, std))
        #     eval_return.append((mean, std))
        # return eval_return
        # except:
        #     self.logger.exception('{}\nGot an error on evaluation.\n{}'.format('!'*50,'!'*50))
        #     raise SystemExit()
            
    def eval_iter(self, _test_target, _pred):
        eval_results_len = len(self.metrics)
        eval_results = np.zeros((1, eval_results_len))
        for i, eval_name in enumerate(self.metrics):
            measure_path = importlib.import_module('task.metric')
            eval = getattr(measure_path, eval_name)
            if self.data_name == 'mg':
                # eval_result = eval(_test_target, _pred, ith = 84)
                eval_result = eval(_test_target, _pred)
            else:
                eval_result = eval(_test_target, _pred)
            self.logger.critical(
                'Testing\t{}:\t{:.4g}'.format(eval_name, eval_result))
            eval_results[0, i] = eval_result
        return eval_results
