import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
class Model(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, N,
                       L,
                       C
                       ):
        super(Model, self).__init__()
        self.seq_len = N
        self.pred_len = L
        self.individual = False#configs.individual
        self.channels = C
        H_order = 8
        self.dominance_freq= 106 #122 #196 #122 #int(N // 96 + 1) * H_order + 10 # 720/24
        # if self.dominance_freq > (self.seq_len / 4) // 2 + 1:
        #     self.dominance_freq_in = int((self.seq_len / 4) // 2 + 1)
        # else: self.dominance_freq_in = self.dominance_freq

        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]


    def forward(self, x, test = False):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.dominance_freq:]=0 # LPF
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
        # low_specx[:,self.dominance_freq_in:]=0 # LPF
        # low_specx = low_specx[:,0:self.dominance_freq_in,:] # LPF

        # if test:
        #     low_specx = low_specx * 4
        # if low_specx.shape[1] != self.dominance_freq:
        #     B, L, C = low_specx.shape
        #     low_specx = torch.concat((low_specx, torch.zeros(B,self.dominance_freq - L, C, dtype=torch.cfloat).to(x.device)), dim = 1)
      
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.dominance_freq*self.length_ratio),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
        # print(low_specxy_)
        low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio # compemsate the length change
        # dom_x=x-low_x
        
        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy=(low_xy) * torch.sqrt(x_var) +x_mean
        return xy, low_xy* torch.sqrt(x_var)

def fits_constructor(hypervar):
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
    
    nfm = Model(N = hypervar.sets_in_training[3],
                L = hypervar.sets_in_training[2],
                C = hypervar.C_true
                )

    model_name = ["FITS"]
    m = [nfm]
    total_p = 0
    for i, name in enumerate(model_name):
        param_ = model_size(m[i], name) if m[i] is not None else 0
        total_p += param_
    logger.info(f"Total parameters: {total_p}")
    logger.info("Model construction was successful")
    return m[0]
