import os, sys
# print(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskLoader import Opt
from task.TaskWrapperV1 import Task
from task.parser import get_parser
from ray import tune
import pandas as pd
from data.simdata import Lorenz_Data as Data
from exp.encoder._model_config import rnn_base


class rnn(rnn_base):
    def task_modify(self):
        self.hyper.epochs = 100

if __name__ == "__main__":
    parser = get_parser(parsing=False)
    args = parser.parse_args()
    
    args.cuda = True
    args.test = True
    # args.clean = False
    args.datafolder = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    args.dataset = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    
    args.exp_name = 'RL'
    args.H = 2
    args.model = 'rnn'
    args.rep_times = 1
    
   
    task = Task(args)
    # task.tuning()
    task.conduct()
    args.metrics = ['rmse','smape', 'nrmse']
    task.evaluation(args.metrics, remove_outlier=False)    