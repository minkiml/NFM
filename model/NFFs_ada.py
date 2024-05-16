import torch.nn as nn
from model.FFs import (FNO, GFN, AFNO, AFF, INFF_ada)
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
            self.NFF_block = INFF_ada(vars=vars, time_= True,
                                  dropout= dropout)
            
        elif self.hyper_vars_NFF.filter_type  == "FNO":# add this filter type to argument TODO
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
                                                 activation = "ReLU2", # "ReLU2"
                                                 type_ = "linear",
                                                 layer_id= layer_id,
                                                 std= self.hyper_vars_NFF.init_std,
                                                 bias = False,
                                                 num_layers = 2,
                                                 init_ = 1)

        self.norm1 = NormalizationLayer(norm = "LayerNorm_feature",  # LayerNorm_feature  LayerNorm_seq
                                        hidden = self.hyper_vars_NFF.hidden_dim, 
                                        affine = True,
                                        dc_removal = False)
        self.norm2 = NormalizationLayer(norm = "LayerNorm_feature", 
                                hidden = self.hyper_vars_NFF.hidden_dim, 
                                affine = True,
                                dc_removal = False)
        self.dropout = nn.Dropout(dropout)

        # self.nff_fl = PositionwiseFeedForward(self.hyper_vars_NFF.hidden_dim, self.hyper_vars_NFF.hidden_dim * 2, 
        #                                     dropout=dropout, 
        #                                     type_ = "linear-c",
        #                                     activation = "GeLU",
        #                                     layer_id= layer_id,
        #                                     std= self.hyper_vars_NFF.init_std)
        
        # self.nff_fl = ForuierChannelMixer(var= vars,
        #                                          hidden = hidden_dim,
        #                                          factor = 3,
        #                                          bias = False,
        #                                          shared= False)

        # self.nff_fl = ForuierChannelMixer2(var= vars,
        #                                          hidden = hidden_dim,
        #                                          factor = 3,
        #                                          bias = True,
        #                                          shared= True,
        #                                          dropout= dropout,
        #                                          nl=False)

        # self.skip_connection = nn.Sequential(
        #                                    nn.Linear(self.hyper_vars_NFF.hidden_dim, int(self.hyper_vars_NFF.hidden_dim / 2), 
        #                         bias = False),
        #                         Sine(),#nn.LeakyReLU(0.2),
        #                         mul_omega(feature_dim = int(self.hyper_vars_NFF.hidden_dim / 2), 
        #                         omega = 1,
        #                         omega_learnable = True,
        #                         gaussian = True),
        #                         nn.Linear(int(self.hyper_vars_NFF.hidden_dim / 2), self.hyper_vars_NFF.hidden_dim, 
        #                         bias = False))
        # for layer in self.skip_connection:
        #     if isinstance(layer, (nn.Linear, nn.Conv1d)):
        #         print("!")
        #         trunc_normal_(layer.weight, std=0.02 if self.hyper_vars_NFF.init_std is None else self.hyper_vars_NFF.init_std)


        # if self.skip_connection.weight is not None:
        #     print("!")
            
        #     trunc_normal_(self.skip_connection.weight, std=0.02 if self.hyper_vars_NFF.init_std is None else self.hyper_vars_NFF.init_std)
        #     self.skip_connection.weight.data.div_(math.sqrt(2.0 * layer_id))
        self.droppath = nn.Identity()
        # if self.hyper_vars_NFF.droppath != 1:
        #     self.droppath = DropPath(p=self.hyper_vars_NFF.droppath)
        # else:
        #     self.droppath = nn.Identity()

    def forward(self, x, xx = None, temporal_loc = None):
        B, L_base, _ = x.shape
        x_ = self.norm1(self.TD_filter_NFF(x))
        # x_ = self.norm1(x)

        freqa = self.hyper_vars_NFF.DFT_(x_)
        # freqa = self.hyper_vars_NFF.care_DC(freqa)
        freqa = freqa[:,:self.hyper_vars_NFF.freq_span,:]

        freq2, freq, td_rep = self.NFF_block(freqa, xx, x_, True, temporal_loc)
        # freq2 = self.hyper_vars_NFF.care_DC(freq2)
  
        z = self.hyper_vars_NFF.IDFT_(freq2, L = self.hyper_vars_NFF.L_span) #+ x
        # z = self.dropout(self.norm2(z + x_))
        # z = (self.nff_fl(self.norm2((z) + x)))
        # z =self.norm2(self.droppath(z)) + self.nff_fl(x_)
        # z =self.norm2(self.droppath(z) + self.nff_fl(x_))
        z = self.norm2(self.droppath(z) + x_)
        # z = self.nff_fl(self.norm2(self.droppath(z) + x_))
        # z = z + self.skip_connection(x)
        # z =self.norm2(self.droppath(z) + x_)
        return z, freq#, freq, x_ #  (self.TD_filter_NFF(z)) + x     self.norm(self.dropout(td) + x)
    


    def forward_insp(self, x, xx = None):
        B, L_base, _ = x.shape

        x_ = self.norm1(self.TD_filter_NFF(x))
        # x_ = self.norm1(x)

        freqa = self.hyper_vars_NFF.DFT_(x_)
        freqa = self.hyper_vars_NFF.care_DC(freqa)

        freqa = freqa[:,:self.hyper_vars_NFF.freq_span,:]

        freq2, freq, td_rep = self.NFF_block(freqa, xx, False)
        freq2 = self.hyper_vars_NFF.care_DC(freq2)
  
        z = self.hyper_vars_NFF.IDFT_(freq2, L = self.hyper_vars_NFF.L_span) #+ x
        # z = self.dropout(self.norm2(z + x_))
        # z = (self.nff_fl(self.norm2((z) + x)))
        # z =self.norm2(self.droppath(z)) + self.nff_fl(x_)
        # z =self.norm2(self.droppath(z) + self.nff_fl(x_))
        z = self.nff_fl(self.norm2(self.droppath(z) + x_))

        # z =self.norm2(self.droppath(z) + x_)
        return z, freq, x_, freq2#, freq, x_ #  (self.TD_filter_NFF(z)) + x     self.norm(self.dropout(td) + x)