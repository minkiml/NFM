import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import os
from model.NFM_backbone import NFM_general
from utils.vars_ import HyperVariables

class NFM_AD(nn.Module):
    '''
    Wrapper class
    '''
    def __init__(self, 
                 var:HyperVariables,
                 mask_ = False,
                 dsr = 1):
        super(NFM_AD, self).__init__()    
        self.hyper_vars = var

        # NFM
        self.NFM_backbone = NFM_general(self.hyper_vars,
                                dropout = self.hyper_vars.dropout)

        # Header
        self.projection_out = nn.Linear(self.hyper_vars.hidden_dim, self.hyper_vars.C_, 
                                            bias = True) 
        self.mask_ = mask_
        if mask_:
            self.masking = torch.ones(self.hyper_vars.L_base)
            self.masking[0::dsr] = 0.
            self.num_masked = len(torch.where(self.masking == 0.)[0])
    def forward(self, x):
        # Batch, Length, original channel dim
        x = self.hyper_vars.input_(x) # Channel independence
        B, L, C = x.shape 
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean   
        x_std=torch.sqrt(torch.var(x, dim=1, keepdim=True)+ 1e-5)
        x = x / x_std
        
        # NFM
        z = self.NFM_backbone(x) # (B, L_base, hidden) 
        
        # Reconstruction
        B, L, c = z.shape
        z_forward = self.projection_out(z)

        z_forward = z_forward * x_std
        z_forward = z_forward +x_mean 
        y, y_freq = self.hyper_vars.output_(z_forward)

        return y, y_freq
    
    def criterion(self, rec_x, y, 
                      x_freq,
                      loss_mode = "TD",
                      reduction = 'mean',
                      mode = "training"):
        if reduction == 'none':
            pass
        if loss_mode == "TFDR" or loss_mode == "TD" or mode == "testing":
            # full pred vs full gt y
            if self.mask_ and mode == "training":
                rec_x = rec_x * self.masking.unsqueeze(0).unsqueeze(-1).to(rec_x.device)
                gt_y = y * self.masking.unsqueeze(0).unsqueeze(-1).to(rec_x.device)
            else: gt_y = y
            TD_loss = F.mse_loss(rec_x, gt_y, reduction = reduction)
        else: TD_loss = torch.tensor(0.)
        if mode == "training":
            if loss_mode == "TFDR" or loss_mode == "FD":
                B, f, C = x_freq.shape
                freq_y = (self.hyper_vars.DFT_(y)[:,:f,:]).detach()

                loss_real = (x_freq.real -  freq_y.real)**2
                loss_imag = (x_freq.imag - freq_y.imag)**2
                FD_loss =  (torch.sqrt(loss_real + loss_imag+ 1e-12))
                FD_loss = FD_loss if reduction == 'none' else FD_loss.mean() 
            else: FD_loss = torch.tensor(0.)
        else: FD_loss = torch.tensor(0.)
        return TD_loss, FD_loss

def model_constructor(hypervar,
                      dsr,
                      masking):
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
    
    nfm = NFM_AD(hypervar,
                 mask_ = masking,
                 dsr = dsr)
    model_name = ["NFM_AD"]
    m = [nfm]
    total_p = 0
    for i, name in enumerate(model_name):
        param_ = model_size(m[i], name) if m[i] is not None else 0
        total_p += param_
    logger.info(f"Total parameters: {total_p}")
    logger.info("Model construction was successful")
    return m[0]