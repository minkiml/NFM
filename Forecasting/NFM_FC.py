import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import os
from model.NFM_backbone import NFM_general

from utils.vars_ import HyperVariables
from Forecasting.predictor import Forecasting_head
class NFM_FC(nn.Module):
    '''
    Wrapper class
    '''
    def __init__(self, 
                 var:HyperVariables):
        super(NFM_FC, self).__init__()    
        self.hyper_vars = var
        
        self.NFM_backbone = NFM_general(self.hyper_vars,
                                dropout = self.hyper_vars.dropout)

        # Header
        self.predictor = Forecasting_head(var)

    def forward(self, x):
    
        x = self.hyper_vars.input_(x) # Channel independence
        # IN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        if self.hyper_vars.norm_trick == "mean_std":
            x_std=torch.sqrt(torch.var(x, dim=1, keepdim=True)+ 1e-5)
            x = x / x_std
        
        # B, L, C = x.shape 
        # NFM backbone
        z= self.NFM_backbone(x)

        # Forecasting header
        y = self.predictor(z)

        if self.hyper_vars.norm_trick == "mean_std":
            y = y * x_std
        y = y +x_mean
        y, y_freq = self.hyper_vars.output_(y)

        # Final outputs
        x_horizon = y[:,-self.hyper_vars.L_out:,:]
        return y, y_freq, x_horizon

    def criterion(self, 
                pred_TD, target_TD, 
                pred_FD, target_FD,
                pred_TD_full, target_TD_full,
                loss_mode = "TFD",
                masking_ = False):
        if loss_mode == "TFD" or loss_mode == "TD":
            # TD loss on only the "forecasted region"
            if masking_:
                mask_ = 1#generate_random_mask(pred_TD.shape, mask_prob=0.3).to(pred_TD.device)
            else: mask_ = 1.
            TD_loss = F.mse_loss(pred_TD * mask_, target_TD * mask_)

        elif loss_mode == "TFDR" or loss_mode == "TDR":
            if masking_:
                mask_ = 1#generate_random_mask(pred_TD_full.shape, mask_prob=0.3).to(pred_TD_full.device)
            else: mask_ = 1.
            TD_loss = F.mse_loss(pred_TD_full * mask_, target_TD_full * mask_)

        else: TD_loss = torch.tensor(0.)

        if loss_mode == "TFD" or loss_mode == "FD" or loss_mode == "TFDR":
            pred_FD = pred_FD[:,:,:]
            target_FD = target_FD[:,:,:]
            # FD loss
            B, f, dim_ = pred_FD.shape
            loss_gamma = 1.0
            assert pred_FD.shape[1] == target_FD.shape[1] and pred_FD.shape[2] == target_FD.shape[2]
            loss_real = (pred_FD.real -  target_FD.real)**2
            loss_imag = (pred_FD.imag - target_FD.imag)**2
            FD_loss =  (torch.sqrt(loss_real + loss_imag+ 1e-12)).mean() 
        else: FD_loss = torch.tensor(0.)
        return TD_loss, FD_loss
    


def model_constructor(hypervar
                      ):
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
    
    nfm = NFM_FC(hypervar)
    model_name = ["NFM_AD"]
    m = [nfm]
    total_p = 0
    for i, name in enumerate(model_name):
        param_ = model_size(m[i], name) if m[i] is not None else 0
        total_p += param_
    logger.info(f"Total parameters: {total_p}")
    logger.info("Model construction was successful")
    return m[0]

