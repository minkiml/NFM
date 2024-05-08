import torch.nn as nn
from utils.vars_ import HyperVariables

class Forecasting_head(nn.Module):
    '''
    Forecasting header d -> c (1 if channel independence)    
    '''
    def __init__(self, 
                 vars: HyperVariables): # Time
        super(Forecasting_head, self).__init__() 
        self.hyper_vars_fore = vars
        self.forecaster_ = nn.Linear(self.hyper_vars_fore.hidden_dim, self.hyper_vars_fore.C_, 
                                    bias = True)
        self.weights_initialization()
    def forward(self, x):
        B, L_base, _ = x.shape
        pred = self.forecaster_(x)
        return pred

    def nn_initialization(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                pass
            if m.bias is not None:
                pass
                m.bias.data.zero_()   
        else: pass

    def weights_initialization(self):
        self.init_mean = 0.
        self.init_std = 0.02
        for var, m in self.named_children():
            if var == "projection_out": 
                self.factor = 0.02
            elif var == "forecaster_":
                self.factor = 0.02
                m.apply(self.nn_initialization)