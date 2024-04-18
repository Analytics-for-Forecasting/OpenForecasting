from task.parser import get_parser
from task.TaskWrapperV1 import Task
# from models.stochastic.cnn import ESM_CNN


if __name__ == "__main__":
    
    args = get_parser()

    task = Task(args)
    # task.tuning()
    task.conduct()

    args.metrics = ['rmse','nrmse', 'mase']
    task.evaluation(args.metrics)
 
 
