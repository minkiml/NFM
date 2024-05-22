import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import os
import math
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
                                        pooling_method = "AVG",
                                        processing ="mean"
                                        )
        self.softmax = nn.Softmax(dim = 1)

        self.smoothing = 0.25
    def forward(self, x):
        x = self.hyper_vars.input_(x)
        B, L, C = x.shape 
        # NFM
        z = self.NFM_backbone(x)
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
        self.smoothing_rate_step()
        return F.cross_entropy(logit, target, label_smoothing=self.smoothing)
    
    def smoothing_rate_step(self):
        if self.hyper_vars.CE_smoothing_scheduler:
            self._step += 1
            if self._step < self.warmup_steps:
                progress = float(self._step) / float(max(1, self.warmup_steps))

                new_rate = self.start_rate + progress * (self.ref_rate - self.start_rate)
            else:
                # -- consine annealing after warmup
                progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
                new_rate = self.final_rate + (self.ref_rate - self.final_rate) * 0.5 * (1. + math.cos(math.pi * progress))
            
            if new_rate > 0.45:
                new_rate = 0.45
                self.hyper_vars.CE_smoothing_scheduler = False
            elif new_rate < 0.:
                new_rate = 0.
                self.hyper_vars.CE_smoothing_scheduler = False
            self.smoothing = new_rate

    def smoothing_rate_reset(self, warmup_steps, start_rate, ref_rate, final_rate, T_max):
        self.start_rate = start_rate
        self.ref_rate = ref_rate
        self.final_rate = final_rate
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

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

