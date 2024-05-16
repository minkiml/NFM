import torch.nn as nn
from model.FFs import (FNO, GFN, AFNO, AFF, INFF)
from model.util_nets import PositionwiseFeedForward
from utils.vars_ import HyperVariables
from model.norms import NormalizationLayer

class mixer_block(nn.Module):
    # Neural Fourier filter layer
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
                            hidden_hidden=hidden_dim * 1,
                            head_num= 2,
                            bias_ = False)
        elif self.hyper_vars_NFF.filter_type  == "GFN":
            self.NFF_block = GFN(hidden_dim,
                                f_modes= self.hyper_vars_NFF.freq_span)
            
        elif self.hyper_vars_NFF.filter_type  == "AFF":
            self.NFF_block = AFF(hidden_dim,
                            hidden_out=hidden_dim * 1,
                            head_num= 2,
                            bias_ = False)
            
        self.TD_filter_NFF = PositionwiseFeedForward(hidden_dim, hidden_dim * hidden_factor, 
                                                 dropout=dropout, 
                                                 activation = "ReLU2",
                                                 type_ = "linear",
                                                 layer_id= layer_id,
                                                 std= self.hyper_vars_NFF.init_std,
                                                 bias = False,
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
        return z, None
    
    def channel_mixing(self, x):
        if self.hyper_vars_NFF.norm_method == "prenorm":
            x = self.norm1(x)
        x = self.TD_filter_NFF(x)
        if self.hyper_vars_NFF.norm_method == "postnorm":
            x = self.norm1(x)
        return x

    def token_mixing(self, x, conditional = None, tl = None):
        residual = x
        if self.hyper_vars_NFF.norm_method == "prenorm":
            x = self.norm2(x)

        freqa = self.hyper_vars_NFF.DFT_(x)
        freqa = freqa[:,:self.hyper_vars_NFF.freq_span,:]
        freq2, freq, td_rep = self.NFF_block(freqa, conditional, tl)  
        x = self.hyper_vars_NFF.IDFT_(freq2, L = self.hyper_vars_NFF.L_span) + residual

        if self.hyper_vars_NFF.norm_method == "postnorm":
            x = self.norm2(x)
        return x