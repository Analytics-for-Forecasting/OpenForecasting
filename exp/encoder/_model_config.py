from models._comSetting import nn_base
from ray import tune

class rnn_base(nn_base):
    def __init__(self):
        self.training = True
        self.arch = 'rnn'
        super().__init__()
        
    def base_modify(self,):
        self.import_path='models/training/RNN.py'
        self.class_name = 'RecurrentNeuralNetwork'
        
    def hyper_init(self,):        
        self.hyper.cell_type = 'lstm'
        # self.hyper.readout_steps = 1 # last states, default 1 (equal to FCD output arch.)
        self.hyper.cov_dim = 0
        self.hyper.learning_rate = 0.01
        self.hyper.step_lr = 10
        self.hyper.epochs = 1000
        self.hyper.hidden_size = 400
