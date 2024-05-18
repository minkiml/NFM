import torch.nn as nn
import torch.nn.init as init
from model.LFT import LFT_block, PositionalEncoding
from model.Mixer_block import mixer_block
from utils.vars_ import HyperVariables
from model.util_nets import (GLU_projection, PositionwiseFeedForward, mul_omega, Periodic_activation)
from model.util_nets import fourier_mapping
class NFM_general(nn.Module):
    ''' 
    General NFM-net 
    '''
    def __init__(self, 
                 vars:HyperVariables,
                 dropout = 0.2):
        super(NFM_general, self).__init__()    
        self.hyper_var_nfm = vars
        self.projection_in = nn.Linear(self.hyper_var_nfm.C_, self.hyper_var_nfm.hidden_dim, 
                                       bias = False)
        self.skip_connection = nn.Sequential(
                                           nn.Linear(self.hyper_var_nfm.C_, self.hyper_var_nfm.hidden_dim * self.hyper_var_nfm.proj_factor, 
                                bias = False),
                                mul_omega(feature_dim = self.hyper_var_nfm.hidden_dim * self.hyper_var_nfm.proj_factor, 
                                omega = 1,
                                omega_learnable = False),
                                Periodic_activation(nl="mix"),#nn.LeakyReLU(0.2),
                                nn.Linear(self.hyper_var_nfm.hidden_dim * self.hyper_var_nfm.proj_factor, self.hyper_var_nfm.hidden_dim, 
                                bias = False)
                                )
        # self.skip_connection = GLU_projection(dim_in=self.hyper_var_nfm.C_,
        #                              out_dim= self.hyper_var_nfm.hidden_dim,
        #                              act = "periodic",
        #                              projection_omega = 1., 
        #                              type_= "linear-c")
        
        self.pos_emb = PositionalEncoding(d_model=self.hyper_var_nfm.hidden_dim, 
                                          max_len=self.hyper_var_nfm.L_span,
                                          scaler= False)
        # LFB layer
        self.LFT_layer = LFT_block(vars = vars,
                              dropout = dropout)
        
        # NFF layer 
        self.mixer_layers = nn.ModuleList([mixer_block(vars = vars,
                                    dropout = dropout,
                                    layer_id= i + 1) for i in range(self.hyper_var_nfm.layer_num)])     

        self.ll_NFF =PositionwiseFeedForward(self.hyper_var_nfm.hidden_dim, self.hyper_var_nfm.hidden_dim, 
                                            dropout=dropout, 
                                            activation = "ReLU2", #"GeLU",
                                            out_dim= self.hyper_var_nfm.hidden_dim,
                                            std= self.hyper_var_nfm.init_std,
                                            bias=False,
                                            init_= 2)
        if self.hyper_var_nfm.tau_in_inrs == "shared":
            self.FF_mapping = fourier_mapping(ff_dim = self.hyper_var_nfm.LFT_siren_hidden,  
                                            ff_sigma=256,
                                            learnable_ff = True,
                                            ff_type = "gaussian", # deterministic_exp  gaussian
                                            L = self.hyper_var_nfm.L_base)
        self.weights_initialization()
    def forward(self, x):
        B, f_in, c = x.shape
        assert c == self.hyper_var_nfm.C_
        x1 = self.projection_in(x)
        x2 = self.skip_connection(x)
        x = x1 + x2

        temporal_loc = self.FF_mapping(L=self.hyper_var_nfm.L_base, dev = x.device) if self.hyper_var_nfm.tau_in_inrs == "shared" else None
        # LFT
        z, f_token = self.LFT_layer(x, temporal_loc = temporal_loc)
        residuel = z
        condition_ = z
        
        #NFF
        z = z + self.pos_emb(B)
        for i, layer in enumerate(self.mixer_layers):
            z, freq = layer(z, condition_, temporal_loc = temporal_loc) 

        z = z + residuel 
        z = self.ll_NFF(z) 
        return z, freq, f_token, z, x
     
    def nn_initialization(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)): 
            if m.weight is not None:
                pass
                print("~~")
                if self.init == 1:
                    # init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    pass
                elif self.init == 0:
                    pass
            if m.bias is not None: # TODO
                if self.init == 3:
                    m.bias.data.zero_()  
                    pass
    def weights_initialization(self):
        for var, m in self.named_children():
            self.init_std = self.hyper_var_nfm.init_std #0.05
            self.init_mean = 0.
            if var == "projection_in" or var == "projection_in1": 
                self.init = 1 # 0.7
                m.apply(self.nn_initialization)
            if var == "projection_in2": 
                self.init = 0 # 0.7
                m.apply(self.nn_initialization)