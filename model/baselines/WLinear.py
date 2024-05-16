import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os

def patchifying(input_tensor, P):
    B, L = input_tensor.size()
    # Reshape the tensor
    grouped_tensor = input_tensor.view(B, -1, P)
    return grouped_tensor

class patch_Linear(nn.Module):
    # Linear layer applied to "patch" --> equivalent to group linear!
    def __init__(self, L, hidden_dim, patch_size = 16):
        super(patch_Linear, self).__init__()
        assert L % patch_size == 0, "L must be divisible by P"
        self.patch_size = patch_size
        self.p_linear = nn.Linear(self.patch_size, hidden_dim)
    def forward(self, x):
        # patchification
        patched_x = patchifying(x, self.patch_size) # B, L/P, P

        x = self.p_linear(patched_x) # (B, L/P, P) -> (B, L/P, h)
        x = torch.flatten(x, 1, -1)
        return x # (B, g * h)
class wLinear(nn.Module):
    def __init__(self, N,
                        L,
                        C):
        super(wLinear, self).__init__()

        self.lookback = N
        self.horizon = L
        self.channel = C
        self.nonlinearity = False
        self.patchification = True
        self.num_layers = 2
        dropout = 0.2
        if not self.patchification:
            in_dim = self.lookback
            out_dim = self.horizon
            self.wlinear =  nn.Sequential(nn.Linear(in_dim, int(in_dim * 5)),
                                          nn.ReLU() if self.nonlinearity else nn.Identity(),
                                          nn.Dropout(p=dropout),
                                          nn.Linear(int(in_dim * 5), out_dim))  
        else:
            patch_size = 16
            in_dim = self.lookback # patch size 
            out_dim = self.horizon
            self.wlinear =  nn.Sequential(patch_Linear(N, 64,
                                                       patch_size=patch_size),
                                            nn.ReLU() if self.nonlinearity else nn.Identity(),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(64 * int(N/patch_size), out_dim))  

    def forward(self, x):
        # x (B, L, C)
        B, L, C = x.shape
        # RevIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        # x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # # print(x_var)
        # x = x / torch.sqrt(x_var)
        # forward process
        y = []
        for i in range (C):
            y.append(self.wlinear(x[:,:,i]).unsqueeze(-1))

        y = torch.concatenate((y), dim = -1)

        # RevIN denorm
        y=y  + x_mean # * torch.sqrt(x_var)
        return y, None
    
def WLinear_constructor(hypervar):
    log_loc = os.environ.get("log_loc")
    root_dir = os.getcwd() 
    logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all.txt'), level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('Model construction')

    def model_size(m, model_name):
        logger.info(f"Model: {model_name}")
        total_param = 0
        for name, param in m.named_parameters():
            num_params = param.numel()
            total_param += num_params
            logger.info(f"{name}: {num_params} parameters")
        
        logger.info(f"Total parameters in {model_name}: {total_param}")
        logger.info("")
        return total_param
    
    nfm = wLinear(N = hypervar.sets_in_training[3],
                L = hypervar.sets_in_training[2],
                C = hypervar.C_true
                )

    model_name = ["WLinear"]
    m = [nfm]
    total_p = 0
    for i, name in enumerate(model_name):
        param_ = model_size(m[i], name) if m[i] is not None else 0
        total_p += param_
    logger.info(f"Total parameters: {total_p}")
    logger.info("Model construction was successful")
    return m[0]
