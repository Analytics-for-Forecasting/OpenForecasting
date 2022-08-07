
from os import write
from task.TaskWrapper import Task
from task.parser import get_parser
# from models.stochastic.cnn import ESM_CNN
import csv

import pandas as pd


if __name__ == "__main__":

    args = get_parser()
    args.test = False
    args.cuda = True
    args.datafolder = 'paper/eto-sdnn'

    args.dataset = 'ablation'
    args.H = 1
    
    # args.dataset = 'narma'
    # args.H = 1
    
    # args.dataset = 'mg'
    # args.H = 84
    
    # args.dataset = 'gef'
    # args.H = 24

    args.rep_times = 15

    args.logger_level = 20

    # model_list = ['esn','gesn','iesn','desn','esm','wider']
    # model_list = ['esn', 'desn', 'gesn', 'iesn', 'pgesn', 'esm', 'wider_pcr']
    model_list = ['wider', 'pt', 'st', 'rt', 'ps', 'pr', 'sr', 'wider_pcr']
    # model_list = ['esn', 'desn', 'gesn', 'esm', 'wider_pcr']
    args.metrics = ['smape','nrmse']
    models = {}

    # init the task with the first model to load the data info:
    args.model = model_list[0]
    task = Task(args)

    for model in model_list:
        args.model = model

        task.model_config(args)
        task.exp_config(args)
        # task.tuning()
        # task.conduct()
        eval_info = task.evaluation(args.metrics)

        models[model] = eval_info

    output_table = []
    header = []
    header.append('{}.H{}'.format(args.dataset, args.H))
    header.extend(model_list)
    output_table.append(header)
    for metric in args.metrics:
        row = [metric]

        model_result = []
        for model in model_list:
            # e = '{:.4g} ({:.3g})'.format(models[model][metric]['mean'], models[model][metric]['std'])
            e = '{:.3e}'.format(models[model].all[metric]['mean'])
            model_result.append(e)
        row.extend(model_result)

        output_table.append(row)

    output_table = pd.DataFrame(output_table).T.values.tolist()

    save_file = '{}/task_eval.{}.H{}.csv'.format(task.exp_dir, args.dataset, args.H)
    with open(save_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output_table)

    if task.data_opts.info.num_series > 1:
        # metric = args.metrics[0]

        for metric in args.metrics:
            output_table = []
            header = []
            header.append('{}.H{}.{}'.format(args.dataset, args.H, metric))
            header.extend(model_list)
            output_table.append(header)

            # header.extend(['sid {}'.format(id) for id in range(task.data_opts.info.num_series)])
            for id in range(task.data_opts.info.num_series):
                row = [str(id)]
                model_result = []
                for model in model_list:
                    e = '{:.3e}'.format(
                        models[model].series[id][metric]['mean'])
                    model_result.append(e)

                row.extend(model_result)
                output_table.append(row)

            output_table = pd.DataFrame(output_table).T.values.tolist()
            save_file = '{}/series_eval.{}.csv'.format(task.exp_dir, metric)
            with open(save_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(output_table)
