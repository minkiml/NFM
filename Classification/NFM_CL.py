import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import os
from model.NFM_backbone import NFM_general
from utils.vars_ import HyperVariables
from Classification.predictor import Classifier

class NFM_CL(nn.Module):
    '''
    Wrapper class
    '''
    def __init__(self, 
                 var:HyperVariables):
        super(NFM_CL, self).__init__()    
        self.hyper_vars = var
        # NFM
        self.NFM_backbone = NFM_general(self.hyper_vars,
                                dropout = self.hyper_vars.dropout)
        # Header
        self.predictor = Classifier(vars = self.hyper_vars,
                                        factor_reduce = 1,
                                        pooling_method = "AVG",
                                        processing ="mean")
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        x = self.hyper_vars.input_(x)
        B, L, C = x.shape 
        # NFM
        z,  freq , f_token, zz, xx= self.NFM_backbone(x)
        # Classification
        y = self.predictor(z) # B, num_class
        return y
    
    def inference(self, x):
        y = self.forward(x)
        y = self.softmax(y)
        y_pred = torch.argmax(y, dim = 1).squeeze(-1)
        return y_pred
    
    def criterion_cl(self, logit, target):
        assert target.dim() == 1 and logit.dim() == 2
        return F.cross_entropy(logit, target, label_smoothing=0.25)
    
    def freeze_layers(self, layers = ["predictor"]):
        for name, param in self.named_parameters():
            if name in layers: 
                param.requires_grad = False
                print(name)
    def melt_layers(self, layers = ["predictor"]):
        if layers is not None:
            for name, param in self.named_parameters():
                if name in layers: 
                    param.requires_grad = True
                    print(name)
        else:
            for name, param in self.named_parameters():
                param.requires_grad = True
def model_constructor(hypervar):
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
    
    nfm = NFM_CL(hypervar)
    model_name = ["NFM_AD"]
    m = [nfm]
    total_p = 0
    for i, name in enumerate(model_name):
        param_ = model_size(m[i], name) if m[i] is not None else 0
        total_p += param_
    logger.info(f"Total parameters: {total_p}")
    logger.info("Model construction was successful")
    return m[0]

