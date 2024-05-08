import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
from model.LFT import LFT_block, PositionalEncoding
from model.mixingblock import NFF_block
from model.utilities import (F_combine, DropPath, Permute, NoiseInjection, trunc_normal_)
from utils.vars_ import HyperVariables
from model.util_nets import (CV_projection, PositionwiseFeedForward, Linear1d, Sine, mul_omega, Periodic_activation)
from model.util_nets import fourier_mapping

class NFM_general(nn.Module):
    ''' 
    General NFM Backbone 
    '''
    def __init__(self, 
                 vars:HyperVariables,
                 dropout = 0.2):
        super(NFM_general, self).__init__()    
        self.hyper_var_nfm = vars

        self.projection_in = nn.Linear(self.hyper_var_nfm.C_, self.hyper_var_nfm.hidden_dim, 
                                       bias = False)
        self.pos_emb = PositionalEncoding(d_model=self.hyper_var_nfm.hidden_dim, 
                                          max_len=self.hyper_var_nfm.L_span,
                                          scaler= False)
        # LFB layer
        self.LFT_layer = LFT_block(vars = vars,
                              dropout = dropout)
        
        # mixing layer 
        self.NFF_layers = nn.ModuleList([NFF_block(vars = vars,
                                    dropout = dropout,
                                    layer_id= i + 1) for i in range(self.hyper_var_nfm.layer_num)])     
        
        # final channel mixing
        self.ll_NFF =PositionwiseFeedForward(self.hyper_var_nfm.hidden_dim, self.hyper_var_nfm.hidden_dim, 
                                            dropout=dropout, 
                                            activation = "ReLU2", #"GeLU",
                                            out_dim= self.hyper_var_nfm.hidden_dim,
                                            std= self.hyper_var_nfm.init_std,
                                            bias=False,
                                            no_init=False,
                                            init_= 2,
                                            factor= 1)
   
        self.ff_emb = nn.Sequential(
                                    Permute((0,2,1)),
                                    Linear1d(self.hyper_var_nfm.C_, self.hyper_var_nfm.hidden_dim*2),
                                    Permute((0,2,1)),
                                    # nn.Linear(self.hyper_var_nfm.C_, self.hyper_var_nfm.hidden_dim*2),
                                    nn.GLU(-1))
        # initial ff emb
        # self.ff_emb = nn.Sequential(
        #                                    nn.Linear(self.hyper_var_nfm.C_, self.hyper_var_nfm.hidden_dim * 3, 
        #                         bias = True),
        #                         mul_omega(feature_dim = self.hyper_var_nfm.hidden_dim * 3, 
        #                         omega = 1,
        #                         omega_learnable = False,
        #                         gaussian = True,
        #                         spectrum = 1),
        #                         Periodic_activation(nl="mix"),#nn.LeakyReLU(0.2),
        #                         nn.Linear(self.hyper_var_nfm.hidden_dim * 3, self.hyper_var_nfm.hidden_dim, 
        #                         bias = False)
        #                         )

        self.FF_mapping = fourier_mapping(ff_dim = self.hyper_var_nfm.LFT_siren_hidden,  
                                            ff_sigma=256, # 256
                                            learnable_ff = False, # !!
                                            ff_type = "gaussian", # deterministic_exp  gaussian
                                            L = self.hyper_var_nfm.L_base)
       
        self.weights_initialization()
        
    def forward(self, x):
        B, f_in, c = x.shape
        assert c == self.hyper_var_nfm.C_
        x1 = self.projection_in(x)
        x2 = self.ff_emb(x)
        x = x1 + x2

        # temporal_loc = self.FF_mapping(L=self.hyper_var_nfm.L_base, dev = x.device)
        z, f_token = self.LFT_layer(x, temporal_loc = None) # out: B, L_base, hidden  , p, px, x_, x_freq

        residuel = z
        condition_ = z
        
        z = z + self.pos_emb(B)
        for i, layer in enumerate(self.NFF_layers):
            z, freq = layer(z, condition_, temporal_loc = None) # out: B, L_base, hidden     # , freq, s
   
        z = z + residuel 
        z = self.ll_NFF(z) 
        return z, freq, f_token, z, x
     
    def nn_initialization(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)): # (nn.Linear, nn.Conv1d)
            if m.weight is not None:
                pass
                if self.init == 1 or self.init == 2:
                    if self.hyper_var_nfm.option_test == 0:
                        print("xavier relu gain")
                        init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    elif self.hyper_var_nfm.option_test == 1:
                        print("xavier tanh gain")
                        init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                    elif self.hyper_var_nfm.option_test == 2:
                        print("xavier leaky_relu gain")
                        init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
                    elif self.hyper_var_nfm.option_test == 3:
                        print("xavier linear gain")
                        init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('linear'))
                    elif self.hyper_var_nfm.option_test == 4:
                        print("xavier 6 gain")
                        init.xavier_normal_(m.weight, gain=np.sqrt(6))
                elif self.init == 0:
                    pass
                # elif self.init == 2:
                #     pass
                elif self.init == 3:
                    init.xavier_uniform_(m.weight)
                    pass
                elif self.init == 4:
                    pass
            if m.bias is not None:
                m.bias.data.zero_()  
                pass
        else: pass

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

            if var == "ll_NFF":
                self.init = 2 
                self.init_std =self.hyper_var_nfm.init_std # 0.7
                m.apply(self.nn_initialization)
                pass
            if var =="conditional_":
                self.init = 3 
                self.init_std =self.hyper_var_nfm.init_std # 0.7
                # m.apply(self.nn_initialization)
            if var =="skip_connection":
                self.init = 4 
                self.init_std =self.hyper_var_nfm.init_std # 0.7
                m.apply(self.nn_initialization)

##################################### For inspection
    # def forward_insp(self, x):
    #     B, f_in, c = x.shape
    #     assert c == self.hyper_var_nfm.C_

    #     x = self.projection_in(x)
        

    #     # LFT
    #     # x = self.revin(x, 'norm')
    #     # z_ = self.hyper_var_nfm.DFT_(x)
    #     z, t_token, f_token = self.LFT_layer.forward_insp(x) # out: B, L_base, hidden  , p, px, x_, x_freq

    #     residuel = z
    #     # condition_ = self.hyper_var_nfm.DFT_(z)
    #     condition_ = z
    #     #NFF
    #     # x = self.revin(x, 'norm')
    #     # z = self.revin(z, 'norm')
    #     for i, layer in enumerate(self.NFF_layers):
    #         z, nff, x_bar,freq2 = layer.forward_insp(z, condition_) # out: B, L_base, hidden     # , freq, s
    #     # z = self.revin(z, 'denorm')
    #     # Final featuer polishing
    #     z_ = z + residuel # self.droppath(z) + residuel
    #     z = self.ll_NFF(z)
    #     # z_ = self.fl(z_)

    #     return z_, nff, x_bar, residuel, t_token, f_token, z, freq2