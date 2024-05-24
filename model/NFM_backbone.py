import torch.nn as nn
import torch.nn.init as init
from model.LFT import LFT_block, PositionalEncoding
from model.Mixer_block import mixer_block
from utils.vars_ import HyperVariables
from model.util_nets import (PositionwiseFeedForward, mul_omega, Periodic_activation)
from model.util_nets import fourier_mapping
class NFM_general(nn.Module):
    ''' 
    General NFM backbone encoder 
    '''
    def __init__(self, 
                 vars:HyperVariables,
                 dropout = 0.2):
        super(NFM_general, self).__init__()    
        self.hyper_var_nfm = vars
        self.projection_in = nn.Linear(self.hyper_var_nfm.C_, self.hyper_var_nfm.hidden_dim, 
                                       bias = False)
        self.projection_in2 = nn.Sequential(
                                           nn.Linear(self.hyper_var_nfm.C_, 
                                                     self.hyper_var_nfm.hidden_dim * self.hyper_var_nfm.proj_factor, 
                                            bias = False),
                                            mul_omega(feature_dim = self.hyper_var_nfm.hidden_dim * self.hyper_var_nfm.proj_factor, 
                                            omega = 1,
                                            omega_learnable = False),
                                            Periodic_activation(nl="mix"),
                                            nn.Linear(self.hyper_var_nfm.hidden_dim * self.hyper_var_nfm.proj_factor, 
                                                    self.hyper_var_nfm.hidden_dim, 
                                            bias = False)
                                            )
        
        self.pos_emb = PositionalEncoding(d_model=self.hyper_var_nfm.hidden_dim, 
                                          max_len=self.hyper_var_nfm.L_span)
        # LFB layer
        self.LFT_layer = LFT_block(vars = vars)
        
        # NFF layer 
        self.mixer_layers = nn.ModuleList([mixer_block(vars = vars,
                                                        dropout = dropout,
                                                        layer_id= i + 1) for i in range(self.hyper_var_nfm.layer_num)])     

        self.ll_NFF =PositionwiseFeedForward(self.hyper_var_nfm.hidden_dim, self.hyper_var_nfm.hidden_dim, 
                                            dropout=dropout, 
                                            activation = "ReLU", 
                                            out_dim= self.hyper_var_nfm.hidden_dim,
                                            std= self.hyper_var_nfm.init_std,
                                            init_= 2)
        if self.hyper_var_nfm.tau_in_inrs == "shared":
            self.FF_mapping = fourier_mapping(ff_dim = self.hyper_var_nfm.LFT_siren_hidden,  
                                            ff_sigma=self.hyper_var_nfm.ff_std,
                                            learnable_ff = True,
                                            ff_type = "gaussian", # deterministic_exp  gaussian
                                            L = self.hyper_var_nfm.L_base)
        self.weights_initialization()
    def forward(self, x):
        B, f_in, c = x.shape
        assert c == self.hyper_var_nfm.C_
        x1 = self.projection_in(x)
        x2 = self.projection_in2(x)
        x = x1 + x2

        temporal_loc = self.FF_mapping(L=self.hyper_var_nfm.L_base, dev = x.device) \
                                if self.hyper_var_nfm.tau_in_inrs == "shared" else None
        # LFT
        z_0 = self.LFT_layer(x, temporal_loc = temporal_loc)
        residuel = z_0
        condition_ = z_0
        
        #Mixing
        z = z_0 + self.pos_emb(B)
        for i, layer in enumerate(self.mixer_layers):
            z = layer(z, condition_, temporal_loc = temporal_loc) 

        z = z + residuel 
        z = self.ll_NFF(z) 
        return z
     
    def nn_initialization(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)): 
            if m.weight is not None:
                if self.hyper_var_nfm.init_xaviar:
                    init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                else: pass
            if m.bias is not None:
                m.bias.data.zero_()  
                    
    def weights_initialization(self):
        for var, m in self.named_children():
            if var == "projection_in":
                m.apply(self.nn_initialization)
