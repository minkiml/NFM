import torch.nn as nn
from model.NFFs import (FNO, GFN, AFNO, AFF, INFF)
from model.util_nets import PositionwiseFeedForward
from utils.vars_ import HyperVariables
from model.norms import NormalizationLayer

class NFF_block(nn.Module):
    # Neural Fourier filter layer
    def __init__(self, 
                 vars: HyperVariables,
                 dropout,
                 layer_id = 1):
        super(NFF_block, self).__init__() 
        self.hyper_vars_NFF = vars
        hidden_dim = self.hyper_vars_NFF.hidden_dim
        hidden_factor = self.hyper_vars_NFF.hidden_factor

        # Nerual Fourier Filter
        if self.hyper_vars_NFF.filter_type == "INFF":
            self.NFF_block = INFF(vars=vars,
                                  dropout= dropout,
                                  layer_id = layer_id)
            
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xx = None, temporal_loc = None):
        B, L_base, _ = x.shape
        x_ = self.norm1(self.TD_filter_NFF(x))

        freqa = self.hyper_vars_NFF.DFT_(x_)
        freqa = freqa[:,:self.hyper_vars_NFF.freq_span,:]
        freq2, freq, td_rep = self.NFF_block(freqa, xx, temporal_loc)  
        z = self.hyper_vars_NFF.IDFT_(freq2, L = self.hyper_vars_NFF.L_span) #+ x
        z = self.norm2(z + x_)
        return z, freq