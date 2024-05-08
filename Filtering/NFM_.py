import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import os
from utils.vars_ import HyperVariables
from model.NFM_backbone import NFM_general
from Classification.predictor import Classifier

class NFM_filter(nn.Module):
    '''
    Wrapper putting all the layers together
    '''
    def __init__(self, 
                 var:HyperVariables):
        super(NFM_filter, self).__init__()    
        self.hyper_vars = var
        # NFM
        self.NFM_backbone = NFM_general(self.hyper_vars,
                                dropout = self.hyper_vars.dropout)
        # Header
        self.predictor = Classifier(vars = self.hyper_vars,
                                        factor_reduce = 1,
                                        pooling_method = "AVG",
                                        processing ="TD")#TD mean
        



        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        # TODO: This is old version.  need to review this 
        # Batch, Length, original channel dim
        x = self.hyper_vars.input_(x, domain_= "time") # Channel independence, Revstat, DFT
        B, L, C = x.shape 

        # Patch embedding (requires input_ argu to be "time") if applicable -- not implemented yet (this needs to be cared in hyper var)
        pass

        # NFM
        z, freq_, s, resd, t_token, f_token, zz, freq2 = self.NFM_backbone.forward_insp(x) # (B, L_base, hidden)
        # Classification
        y = self.predictor(z) # B, num_class
    
        return y, z , freq_, s, resd, t_token, f_token, zz, freq2
    
    def inference(self, x):
        y, z , freq_, s = self.softmax(self.forward(x))
        y_pred = torch.argmax(y, dim = 1).squeeze(-1)
        return y_pred
    
    def criteria_cla(self, logit, target):

        assert target.dim() == 1 and logit.dim() == 2
        return F.cross_entropy(logit, target)
    
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
    
    nfm = NFM_filter(hypervar)
    model_name = ["NFM_AD"]
    m = [nfm]
    total_p = 0
    for i, name in enumerate(model_name):
        param_ = model_size(m[i], name) if m[i] is not None else 0
        total_p += param_
    logger.info(f"Total parameters: {total_p}")
    logger.info("Model construction was successful")
    return m[0]

