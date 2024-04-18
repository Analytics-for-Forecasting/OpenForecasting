from models._comSetting import nn_base
from task.TaskLoader import Opt
from ray import tune

class esn_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'rnn'
        super().__init__()
        
    def base_modify(self,):
        self.import_path='models/stochastic/esn/ESN.py'
        self.class_name = 'EchoStateNetwork'
        
    def hyper_init(self,):        
        self.hyper.leaky_r = 1
        self.hyper.readout_steps = 1 # last states, default 1 (equal to FCD output arch.)
        self.hyper.hidden_size = 400
        self.hyper.reg_lambda = 0
        self.hyper.nonlinearity = 'tanh'
        self.hyper.iw_bound = (-0.1, 0.1)
        self.hyper.hw_bound = (-1, 1)
        self.hyper.weight_scaling = 0.9
        self.hyper.init = 'vanilla'
        self.hyper.fc_io = 'step'
        self.hyper.input_dim = 1
    
    def tuner_init(self):
        self.tuner.resource = {
            "cpu": 10,
            "gpu": 1  # set this for GPUs
        }
        self.tuner.num_samples = 20
        self.tuning.iw_bound = tune.loguniform(1e-5, 1e-1)
        self.tuning.weight_scaling = tune.uniform(0.2, 0.99)
        self.tuning.nonlinearity = tune.choice(['sigmoid', 'tanh'])        



class bsa_esn_base(esn_base):
    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/weightOptimization/bsaESN.py'
        self.class_name = 'BSA_ESN'
    
class t_esn_base(esn_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/weightOptimization/tESN.py'
        self.class_name = 'TESN'
        
    
class ar_esn_base(esn_base):
    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/ESN.py'
        self.class_name = 'SSO_ESN'

class rd_esn_base(esn_base):
    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/ESN.py'
        self.class_name = 'S2S_ESN'

class sm_esn_base(nn_base):
    '''Has been changed, non-worked! 
    ToDO: See exp/sf/na.py with ss_sho for repair!
    '''
    def __init__(self):
        super().__init__()
        self.training = False
        self.arch = 'rnn'
        self.innerTuning = True
    
    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/StripESN.py'
        self.class_name = 'StateSelection'
        
    def tuner_init(self):
        self.tuner.num_samples = 200
        self.tuner.algo = 'pso'
        self.tuner.resource = {
            'cpu': 5, 'gpu': 0.5
        }
        
class gesn_base(esn_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/GrowingESN.py'
        self.class_name = 'Growing_ESN'
    
    def hyper_modify(self):
        self.hyper.hidden_size = 100
        self.hyper.branch_size = 7
        self.hyper.weight_scaling = 0.9
        self.hyper.hw_bound = (0.66, 0.99)
        self.hyper.nonlinearity = 'sigmoid'
        
class desn_base(esn_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/DeepESN.py'
        self.class_name = 'Deep_ESN'
    
    def hyper_modify(self):
        self.hyper.leaky_r = 0.55
        self.hyper.nonlinearity = 'tanh'
        self.hyper.hidden_size = 100
        self.hyper.num_layers = 10
        
class bpsoesn_base(nn_base):
    def __init__(self):
        super().__init__()
        self.training = False
        self.arch = 'rnn'
        self.innerTuning = True
            
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/weightOptimization/BpsoESN.py'
        self.class_name = 'BPSO_ESN'

    def hyper_init(self):
        self.hyper.esn_opts=esn_base()
        self.hyper.esn_opts.import_path = 'models/stochastic/esn/weightOptimization/BpsoESN.py'
        self.hyper.esn_opts.class_name = 'BpsoESNind'
        self.hyper.esn_opts.common_process()
    
    # def tuner_init(self):
    #     self.tuner.num_samples = 400
    #     self.tuner.algo = 'pso'
    #     self.tuner.resource = {
    #         'cpu': 30, 'gpu': 1
    #     }
        
class bgwoesn_base(nn_base):
    def __init__(self):
        super().__init__()
        self.training = False
        self.arch = 'rnn'
        self.innerTuning = True
            
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/weightOptimization/BgwoESN.py'
        self.class_name = 'BGWO_ESN'
    
    def hyper_init(self):
        self.hyper.esn_opts=esn_base()
        self.hyper.esn_opts.import_path = 'models/stochastic/esn/weightOptimization/BpsoESN.py'
        self.hyper.esn_opts.class_name = 'BpsoESNind'
        self.hyper.esn_opts.common_process()
    
class psb_esn_base(nn_base):
    def __init__(self):
        super().__init__()
        self.training = False
        self.arch = 'rnn'
        self.innerTuning = True
            
    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/weightOptimization/psbESN.py'
        self.class_name = 'PSB_ESN'
        
    def hyper_init(self):
        self.hyper.esn_opts=esn_base()
        self.hyper.esn_opts.import_path = 'models/stochastic/esn/weightOptimization/MESN.py'
        self.hyper.esn_opts.class_name = 'MaskESN'
        self.hyper.esn_opts.common_process()
        
        self.hyper.pop_size = 80
        self.hyper.maxCycle = 100
        self.hyper.bs_epochs = 20      
        # self.tuning.iw_bound = tune.loguniform(1e-5, 1e-1)
        # self.tuning.weight_scaling = tune.uniform(0.2, 0.99)
        # self.tuning.nonlinearity = tune.choice(['sigmoid', 'tanh']) 

class dresn_base(nn_base):
    def __init__(self):
        super().__init__()
        self.training = False
        self.arch = 'rnn'
        # self.innerTuning = True
    
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/weightOptimization/DRESN.py'
        self.class_name = 'DRESN'
        
    def hyper_init(self):
        self.hyper.leaky_r = 1
        self.hyper.readout_steps = 1 # last states, default 1 (equal to FCD output arch.)
        self.hyper.reg_lambda = 0.00001
        self.hyper.nonlinearity = 'sigmoid'
        self.hyper.init = 'svd'
        self.hyper.fc_io = 'off'
        self.hyper.input_dim = 1
        self.hyper.hidden_size = 25
        self.hyper.branch_size = 100
        self.hyper.weight_scaling = 0.9
        self.hyper.iw_bound = (-0.1, 0.1)
        self.hyper.hw_bound = (0.1, 0.99)
        self.hyper.Ng1 = 25
        self.hyper.Ng2 = 5

    def tuner_init(self):
        self.tuner.resource = {
            "cpu": 30,
            "gpu": 1  # set this for GPUs
        }
        self.tuner.num_samples = 1 # set as 1 to enable grid search in randn algo.
        self.tuner.algo = 'grid'
        self.tuning.reg_lambda = tune.grid_search([1e-9, 1e-7,1e-5,1e-3,1e-1,1e1,1e3,1e5,1e7,1e9])
        self.tuning.Ng1=tune.grid_search([5,10,15,20,25])
        self.tuning.Ng2=tune.grid_search([3,5,7,9,11])
     
class psogesn_base(gesn_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/weightOptimization/psoGESN.py'
        self.class_name = 'PSOGESN'
        self.innerTuning = True
    
    def ablation_modify(self,):
        self.hyper.hw_bound = (0.6, 0.99)
        self.hyper.hidden_size = 100
        self.hyper.branch_size = 7
        self.hyper.pso_c1 = 0.01
        self.hyper.pso_c2 = 0.1
        self.hyper.pso_w = 0.12
        self.hyper.pso_iters = 10
        self.hyper.pso_pops = 10
        
class aeesn_base(esn_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/esn/weightOptimization/AEESN.py'
        self.class_name = 'AdaptiveElasticESN'
        # self.innerTuning = True
    
    def tuner_init(self):
        self.tuner.resource = {
            "cpu": 10,
            "gpu": 1  # set this for GPUs
        }
        self.tuner.num_samples = 20
        self.tuning.iw_bound = tune.loguniform(1e-5, 1e-1)
        self.tuning.weight_scaling = tune.uniform(0.2, 0.99)
        self.tuning.reg_alpha = tune.loguniform(1e-2, 1e2)
        self.tuning.l1_ratio = tune.uniform(0.01, 0.99)
        
    def ablation_modify(self,):
        self.hyper.hw_bound = (-0.01, 0.01)
        self.hyper.hidden_size = 400
        self.hyper.reg_alpha = 1
        self.hyper.l1_ratio = 0.5

