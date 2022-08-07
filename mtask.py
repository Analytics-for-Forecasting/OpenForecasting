from task.parser import get_parser
from task.TaskWrapper import Task
# from models.stochastic.cnn import ESM_CNN


if __name__ == "__main__":
    
    args = get_parser()

    args.cuda = True
    args.datafolder = 'paper/eto-sdnn'
    
    args.dataset = 'narma'
    args.H = 1
    args.model = 'esm'
    
    args.rep_times = 1
    args.test = True
    
    task = Task(args)
    # task.tuning()
    task.conduct()

    args.metrics = ['rmse','nrmse', 'mase']
    task.evaluation(args.metrics)
 
 
