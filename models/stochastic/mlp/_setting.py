from models._comSetting import mlp_base
from task.TaskLoader import Opt
from ray import tune

class scn_base(mlp_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/mlp/SCN.py'
        self.class_name = 'StochasticConfigurationNetwork'
        self.innerTuning = True
        self.training = False
        self.arch = 'mlp'
    
class ielm_base(mlp_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/mlp/IELM.py'
        self.class_name = 'IELM'
        self.innerTuning = True
        self.training = False
        self.arch = 'mlp'
            