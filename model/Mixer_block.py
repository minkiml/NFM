import torch.nn as nn
from model.FFs import (FNO, GFN, AFNO, AFF, INFF)
from model.util_nets import PositionwiseFeedForward
from utils.vars_ import HyperVariables
from model.norms import NormalizationLayer

class mixer_block(nn.Module):
    '''
    MAIN mixing block 
    Conventional FFN (channel mixing) + NFF (token mixing)
    '''
    def __init__(self, 
                 vars: HyperVariables,
                 dropout,
                 layer_id = 1):
        super(mixer_block, self).__init__() 
        self.hyper_vars_NFF = vars
        hidden_dim = self.hyper_vars_NFF.hidden_dim
        hidden_factor = self.hyper_vars_NFF.hidden_factor

        # Nerual Fourier Filter
        if self.hyper_vars_NFF.filter_type == "INFF":
            self.NFF_block = INFF(vars=vars,
                                  dropout= dropout)
            
        elif self.hyper_vars_NFF.filter_type  == "FNO":
            self.NFF_block = FNO(hidden_dim,
                                 hidden_dim,
                                 f_modes = self.hyper_vars_NFF.freq_span)
        elif self.hyper_vars_NFF.filter_type  == "AFNO":
            self.NFF_block = AFNO(hidden_dim,
                            hidden_hidden=hidden_dim,
                            head_num= 2,
                            bias_ = False)
        elif self.hyper_vars_NFF.filter_type  == "GFN":
            self.NFF_block = GFN(hidden_dim,
                                f_modes= self.hyper_vars_NFF.freq_span)
            
        elif self.hyper_vars_NFF.filter_type  == "AFF":
            self.NFF_block = AFF(hidden_dim,
                            hidden_out=hidden_dim,
                            head_num= 2,
                            bias_ = False)
            
        # FFN chnnel mixing
        self.channel_mixer = PositionwiseFeedForward(hidden_dim, hidden_dim * hidden_factor, 
                                                 dropout=dropout, 
                                                 activation = "ReLU", 
                                                 type_ = "linear",
                                                 n_slope = 0.04,
                                                 layer_id= layer_id,
                                                 std= self.hyper_vars_NFF.init_std,
                                                 init_ = 1)
        self.norm1 = NormalizationLayer(norm = "LayerNorm", 
                                        hidden = self.hyper_vars_NFF.hidden_dim, 
                                        affine = True)
        self.norm2 = NormalizationLayer(norm = "LayerNorm", 
                                hidden = self.hyper_vars_NFF.hidden_dim, 
                                affine = True)
        
    def forward(self, x, z_0 = None, temporal_loc = None):
        B, L_base, _ = x.shape
        if self.hyper_vars_NFF.mixing_method == "pre_channelmixing":
            z = self.token_mixing(self.channel_mixing(x), z_0, temporal_loc)
        else:
            z = self.channel_mixing(self.token_mixing(x, z_0, temporal_loc))
        return z
    
    def channel_mixing(self, x):
        if self.hyper_vars_NFF.norm_method == "prenorm":
            x = self.norm1(x)
        x = self.channel_mixer(x)
        if self.hyper_vars_NFF.norm_method == "postnorm":
            x = self.norm1(x)
        return x

    def token_mixing(self, x, conditional = None, tl = None):
        residual = x
        if self.hyper_vars_NFF.norm_method == "prenorm":
            x = self.norm2(x)

        x = self.hyper_vars_NFF.DFT_(x)
        x = x[:,:self.hyper_vars_NFF.freq_span,:]
        x = self.NFF_block(x, conditional, tl)  
        x = self.hyper_vars_NFF.IDFT_(x, L = self.hyper_vars_NFF.L_span) + residual

        if self.hyper_vars_NFF.norm_method == "postnorm":
            x = self.norm2(x)
        return x