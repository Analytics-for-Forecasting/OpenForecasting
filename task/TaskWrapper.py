import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import math
from tqdm.std import tqdm
import importlib
from task.util import plot_xfit
from task.util import os_makedirs, os_rmdirs, set_logger
from task.dataset import de_scale
from task.TaskTuner import StocHyperTuner as HyperTuner
from task.TaskLoader import Opt
import torch
import numpy as np
import statistics
from tqdm import trange
import shutil

# from numpy.lib.function_base import select
# from ray.tune import logger
# from task.metric import rmse, mape, smape


class Task(Opt):
    def __init__(self, args):
        # self.opts = Opt()
        # self.opts.merge(args)

        self.exp_module_path = importlib.import_module('data.{}.{}'.format(
            args.datafolder.replace('/', '.'), args.dataset))  # 引入配置
        
        self.data_config(args)
        self.model_config(args)
        self.exp_config(args)
        self.data_subconfig()

    def data_config(self, args):
        self.data_name = args.dataset
        data_opts = getattr(self.exp_module_path, args.dataset + '_data')
        self.data_opts = data_opts(args)

    def data_subconfig(self,):
        self.data_opts.arch = self.model_opts.arch
        self.data_opts.sub_config()

    def model_config(self, args):
        self.model_name = args.model

        # load the specifical config firstly, if not exists, load the common config
        if hasattr(self.exp_module_path, self.model_name):
            model_opts = getattr(self.exp_module_path,
                                 args.model)
        else:
            try:
                share_module_path = importlib.import_module('data.base')
                model_opts = getattr(
                    share_module_path, self.model_name + '_default')
            except:
                raise ValueError(
                    'Non-supported model {} in the data.base module, please check the module or the model name'.format(self.model_name))

        self.model_opts = model_opts()
        self.model_opts.hyper.merge(opts=self.data_opts.info)
        self.model_opts.hyper.H = args.H

        if self.model_opts.arch == 'cnn':
            if not self.model_name == 'clstm':
                self.model_opts.hyper.kernel_size = math.ceil(
                    self.model_opts.hyper.steps / 4)

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

        # if 'statistic' in vars(self.model_opts):
        if self.model_opts.arch == 'statistic':
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

        self.sid_list = args.sid
        self.cid_list = args.cid
        if self.sid_list == ['all']:
            self.sid_list = list(range((self.data_opts.info.num_series)))
        else:
            _temp = [int(s) for s in self.sid_list]
            self.sid_list = _temp
            
        if self.cid_list == ['all']:
            self.cid_list = list(range(self.rep_times))
        else:
            _temp = [int(c) for c in self.cid_list]
            self.cid_list = _temp

        if args.test and args.clean:
            os_rmdirs(self.task_dir)
        os_makedirs(self.task_dir)

        self.model_opts.hyper.device = self.device
        self.tune = args.tune  # default False

    def logger_config(self, dir, stage, cv, sub_count):
        log_path = os.path.join(dir, 'logs',
                                '{}.cv{}.series{}.log'.format(stage, cv, sub_count))
        log_name = '{}.series{}.cv{}.{}'.format(
            self.data_name, sub_count, cv, self.model_name)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger

    def conduct(self,):
        # init and mkdir taskdir
        # generate the subPack dataset
        if self.tune:
            self.tuning()

        # for sub_count, series_Pack in enumerate(tqdm()):
        for series_Pack in tqdm([self.data_opts.seriesPack[s] for s in self.sid_list]):

            # self.task_dir = os.path.join(self.task_dir, 'series_{}'.format(sub_count))
            sub_count = series_Pack.index
            self.series_dir = os.path.join(
                self.task_dir, 'series{}'.format(sub_count))
            self.measure_dir = os.path.join(self.series_dir, 'eval_results')
            os_makedirs(self.measure_dir)

            for i in tqdm(self.cid_list):

                result_file = os.path.join(
                    self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))

                if os.path.exists(result_file):
                    continue
                # if i > 0 and 'statistic' in self.model_opts.dict:
                if i > 0 and   self.model_opts.arch == 'statistic':
                    # assert self.model_opts.statistic
                    result0 = str(os.path.join(
                        self.measure_dir, 'results_{}.series_{}.npy'.format(0, sub_count)))
                    shutil.copy(result0, result_file)
                    continue

                # loading the best paramters:
                cLogger = self.logger_config(
                    self.series_dir, 'train', i, sub_count)
                cLogger.critical('*'*80)
                cLogger.critical('Dataset: {}\t Model:{} \t H: {}\t Trail: {}'.format(
                    self.data_name, self.model_name, self.model_opts.hyper.H, i))

                fit_info = self.conduct_iter(
                    i, series_Pack, result_file, cLogger, innerSaving=True)
                self.plot_fitInfo(fit_info, subId=series_Pack.index,
                                  cvId=i)

    def conduct_iter(self, i, subPack, result_file, clogger, innerSaving=True):
        try:
            if self.tune:
                best_hyper = self.load_tuning(subPack)
                self.model_opts.hyper.update(best_hyper)
                clogger.info("Updating tuning result complete.")
                clogger.critical('-'*80)

            # self.seed = i
            # torch.manual_seed(self.seed)
            # np.random.seed(self.seed)

            clogger.critical(
                'For {}th-batch-trainingLoading, loading the sub-datasets {}'.format(i, subPack.index))
            clogger.critical('-'*80)

            # Attention! sub.H can be different among multiple series in some cases.
            self.model_opts.hyper.H = subPack.H
            self.model_opts.hyper.series_dir = self.series_dir
            self.model_opts.hyper.sid = subPack.index
            self.model_opts.hyper.cid = i
            model = self.model_import()
            model = model(self.model_opts.hyper, clogger)

            clogger.critical('Loading complete.')
            clogger.critical(f'Model: \n{str(model)}')

            fit_info = model.xfit(subPack.train_loader, subPack.valid_loader)
            _, tgt, pred = model.loader_pred(subPack.test_loader)

            _tgt, _pred = de_scale(
                subPack, tgt), de_scale(subPack, pred)

            # _dataset = subPack.orig
            # _orig_tgt = _dataset[-_tgt.shape[0]:, -subPack.H:]
            # diff = rmse(_orig_tgt, _tgt)
            # print(diff)
            # to do, including differential dataset.

            clogger.critical('-'*50)
            if innerSaving:
                np.save(result_file,
                        (_tgt, _pred))
                return fit_info
            else:
                return _tgt, _pred, fit_info
        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def tuning(self,):
        try:
            self.tune = True

            # check tune condition, if not satisfying, jump tune.
            if 'innerTuning' in self.model_opts.dict:
                if self.model_opts.innerTuning == True:
                    self.tune = False

            if self.tune:
                for sub_count, series_Pack in enumerate(tqdm(self.data_opts.seriesPack)):
                    assert sub_count == series_Pack.index

                    self.series_dir = os.path.join(
                        self.task_dir, 'series{}'.format(sub_count))

                    tuner_dir = os.path.join(self.series_dir, 'tuner')
                    os_makedirs(tuner_dir)
                    tuner_path = os.path.join(
                        tuner_dir, 'series{}.best.pt'.format(sub_count))

                    self.model_opts.tuner.dir = tuner_dir

                    tLogger = self.logger_config(
                        tuner_dir, 'tuning', 'T', sub_count)
                    if not os.path.exists(tuner_path):
                        series_tuner = HyperTuner(
                            self.model_opts, tLogger, series_Pack)
                        series_tuner.conduct()
                        best_hyper = series_tuner.best_config
                        tLogger.critical('-'*80)
                        tLogger.critical('Tuning complete.')
                        torch.save(best_hyper, tuner_path)
                    else:
                        best_hyper = torch.load(tuner_path)

                    # best_hyper is a dict type
                    for (arg, value) in best_hyper.items():
                        tLogger.info("Tuning Results:\t %s - %r", arg, value)
            return self.tune

        except:
            tLogger.exception(
                '{}\nGot an error on tuning.\n{}'.format('!'*50, '!'*50))

    def load_tuning(self, subPack):
        tuner_dir = os.path.join(self.series_dir, 'tuner')
        tuner_path = os.path.join(
            tuner_dir, 'series{}.best.pt'.format(subPack.index))
        best_hyper = torch.load(tuner_path)
        if not os.path.exists(tuner_path):
            raise ValueError(
                'Invalid tuner path: {}'.format(tuner_path))
        return best_hyper

    def plot_fitInfo(self, fit_info, subId, cvId):
        if 'loss_list' in fit_info.dict and 'vloss_list' in fit_info.dict:
            plot_dir = os.path.join(self.series_dir, 'figures')
            os_makedirs(plot_dir)

            plot_xfit(fit_info, 'cv{}.series{}'.format(
                cvId, subId), plot_dir)

    def evaluation(self, metrics=['rmse'], force_update=True):
        try:
            self.metrics = metrics
            eval_list = []
            eLogger = set_logger(os.path.join(self.task_dir, 'eval.log'), '{}.H{}.{}'.format(
                self.data_name, self.data_opts.info.H, self.model_name.upper()), self.logger_level)

            for sub_count in self.sid_list:

                ser_eval = []
                self.series_dir = os.path.join(
                    self.task_dir, 'series{}'.format(sub_count))
                self.measure_dir = os.path.join(
                    self.series_dir, 'eval_results')
                os_makedirs(self.measure_dir)

                # check the mase:
                # if 'mase' in self.metrics:
                #     naivePred = self.get_naivePred(sub_count)
                #     self.data_opts.seriesPack[sub_count].naiveP = naivePred

                for i in self.cid_list:
                    metric_file = os.path.join(
                        self.measure_dir, 'metrics_{}.series_{}.npy'.format(i, sub_count))

                    if os.path.exists(metric_file) and force_update is False:
                        eval_results = np.load(metric_file)
                    else:
                        eval_results = self.eval_iter(
                            i, sub_count)

                    eLogger.critical('*'*80)
                    eLogger.critical('Dataset: {}\t Model: {}\t H: {}\tSeries-id: {}\t Trail-id: {}'.format(
                        self.data_name, self.model_name, self.data_opts.info.H, sub_count, i))
                    for _i, eval_name in enumerate(self.metrics):
                        eLogger.critical(
                            'Testing\t{}:\t{:.4g}'.format(eval_name, eval_results[0, _i]))
                    ser_eval.append(eval_results)
                    np.save(metric_file, eval_results)
                eval_list.append(ser_eval)
                eLogger.critical('-'*80)

            self.eval_info = Opt()
            self.eval_info.series = []
            # if self.rep_times > 1:
            for sub_count, ser_eval in enumerate(eval_list):
                eLogger.critical('='*80)
                eLogger.critical('Dataset: {}\t Model: {}\t H: {}\t Series-id: {} \t Trail-Nums: {}'.format(
                    self.data_name, self.model_name, self.data_opts.info.H, sub_count, self.rep_times))

                series_eval_dict = self.eval_list2dict(ser_eval)
                self.eval_info.series.append(series_eval_dict)
                for metric_name in self.metrics:
                    eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
                        metric_name, series_eval_dict[metric_name]['mean'], series_eval_dict[metric_name]['std']))

            eLogger.critical('@'*80)
            eLogger.critical('Dataset: {}\t Model: {}\t H: {}\t Series-Nums: {}\t Trail-Nums: {}'.format(
                self.data_name, self.model_name, self.data_opts.info.H, len(self.sid_list), len(self.cid_list)))

            all_eval_list = [item for series in eval_list for item in series]
            all_eval_avg = self.eval_list2dict(all_eval_list)
            for metric_name in self.metrics:
                eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
                    metric_name, all_eval_avg[metric_name]['mean'], all_eval_avg[metric_name]['std']))

            self.eval_info.all = all_eval_avg

            return self.eval_info
        except:
            eLogger.exception(
                '{}\nGot an error on evaluation.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def eval_iter(self, i, sub_count, result_zip=None):

        if result_zip is not None:
            _test_target, _pred = result_zip[0], result_zip[1]
        else:
            result_file = os.path.join(
                self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))
            _test_target, _pred = np.load(result_file)

        eval_results_len = len(self.metrics)
        eval_results = np.zeros((1, eval_results_len))
        for i, eval_name in enumerate(self.metrics):
            measure_path = importlib.import_module('task.metric')
            eval = getattr(measure_path, eval_name)
            # if self.data_name == 'mg':
            #     # eval_result = eval(_test_target, _pred, ith = 84)
            #     eval_result = eval(_test_target, _pred)
            # else:
            if eval_name == 'mase':
                # naivePred = self.get_naivePred(sub_count)
                eval_result = eval(_test_target, _pred,
                                   self.data_opts.seriesPack[sub_count].avgNaiveError)
            else:
                eval_result = eval(_test_target, _pred)
            eval_results[0, i] = eval_result
        return eval_results

    def get_naivePred(self, subcount):
        subPack = self.data_opts.seriesPack[subcount]
        testloader = subPack.test_loader

        tx = []
        for batch_x, _ in testloader:
            tx.append(batch_x)
        tx = torch.cat(tx, dim=0).detach().cpu().numpy()
        tx = tx[:, 0, :]

        _tx = de_scale(subPack, tx, tag='input')

        _pred = _tx[:, -1]
        return _pred

    def eval_list2dict(self, _eval_list):
        eval_data = np.concatenate(_eval_list, axis=0)

        all_eval_avg = {}
        for i, metric_name in enumerate(self.metrics):
            i_data = eval_data[:, i].tolist()

            if self.rep_times * self.data_opts.info.num_series > 1:
                mean = statistics.mean(i_data)
                std = statistics.stdev(i_data, mean)
            else:
                mean = i_data[0]
                std = 0

            # eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
            #     metric_name, mean, std))

            all_eval_avg[metric_name] = {}
            all_eval_avg[metric_name]['mean'] = mean
            all_eval_avg[metric_name]['std'] = std
            all_eval_avg[metric_name]['raw'] = i_data

        return all_eval_avg
