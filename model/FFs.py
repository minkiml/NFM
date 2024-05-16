
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.init as init
import numpy as np

from model.utilities import trunc_normal_, F_combine
from utils.vars_ import HyperVariables
from model.norms import NormalizationLayer
from model.util_nets import (Siren_block, CV_projection,
                             PositionwiseFeedForward, Sine)
from model.LFT import lft_scale_bias
def complex_mul( x, weights):
    if weights.dim() ==2:
        return torch.einsum("...i,io->...o", x, weights)
    elif weights.dim() == 3:
        return torch.einsum("...fi,fio->...fo", x, weights)

class FNO(nn.Module):
    '''
    "Fourier Neural Operator for Parametric Partial Differential Equations"
    W_i(F_i)
    Input: Frequency rep (B, modes, features)
    output: Filtered Frequency rep (B, modes, features_out)
    Note:
    1) Huge parameter size depending on the hidden size and the frequency modes
    2) Dependent on the number of modes (unless modes are truncated)
    3) The output dimension must be handled with additional neural block (out dim is hidden_out!)
    '''
    def __init__(self, hidden_in = 32, 
                 hidden_out = 128,
                 f_modes = -1,
                 mode_ = 1,
                 init_std = 0.04):
        super(FNO, self).__init__() 
        self.frequency_modes = f_modes
        self.mode_ = mode_

        if self.mode_ == 1: # Complex valued weights (original implementation)
            real_ = torch.zeros(self.frequency_modes, hidden_in, hidden_out)
            imag_ = torch.zeros(self.frequency_modes, hidden_in, hidden_out)
            trunc_normal_(real_, mean = 0., std = init_std, a = -0.2, b = 0.2)
            trunc_normal_(imag_, mean = 0., std = init_std, a = -0.2, b = 0.2)
            self.FNO = nn.Parameter(torch.complex(real_, imag_))

        elif self.mode_ == 2: # Independent real and imaginary
            self.FNO_real = nn.Parameter(torch.zeros(self.frequency_modes, hidden_in, hidden_out))
            self.FNO_imag = nn.Parameter(torch.zeros(self.frequency_modes, hidden_in, hidden_out))
            trunc_normal_(self.FNO_real, mean = 0., std = init_std, a = -0.2, b = 0.2)
            trunc_normal_(self.FNO_imag, mean = 0., std = init_std, a = -0.2, b = 0.2)
    def forward(self, x, n1 = None, n2 = None, n3 = None, n4 = None):
        B, F_, d_ = x.shape
        assert self.frequency_modes == F_
        if self.mode_ == 1:
            return complex_mul(x, self.FNO), None, None
        elif self.mode_ == 2:
            freq_real = complex_mul(x.real, self.FNO_real) - complex_mul(x.imag, self.FNO_imag)
            freq_imag = complex_mul(x.imag, self.FNO_real) + complex_mul(x.real, self.FNO_imag)
            return F_combine(freq_real, freq_imag), None, None

class GFN(nn.Module):
    '''
    "Global filter network"

    Hammarad product (F_i * R_i) which is the same as FNO W_i(F_i) but W_i is diagonal matrix (i.e., no channel mixing)
    
    Note:
    Dependent on the number of modes (unless modes are truncated)
    '''
    def __init__(self, hidden_dim = 32, 
                f_modes = 100,
                 ):
        super(GFN, self).__init__()
        self.frequency_modes = f_modes

        real_ = torch.randn(1,self.frequency_modes, hidden_dim) * 0.02
        imag_ = torch.randn(1,self.frequency_modes, hidden_dim) * 0.02
        self.GF_weights = nn.Parameter(torch.complex(real_, imag_))

    def forward(self, x, n1 = None, n2 = None, n3 = None, n4 = None):
        # n1 and n2 are null arguments
        B, F_, d_ = x.shape
        return x * self.GF_weights, self.GF_weights.detach(), None

class AFNO(nn.Module):
    '''
    "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers"
    W_2(NL(W_1(F_i) + b_1)) + b_2 

    Input: Frequency rep (B, modes, features)
    output: Filtered Frequency rep (B, modes, features_out)

    original implemtation is with mode 2

    Note:
    Completely independent of the number of modes
    '''
    def __init__(self, hidden = 32, hidden_hidden = 128,
                 head_num = 4, bias_ = False,
                 adaptive = False):
        super(AFNO, self).__init__()
        self.hidden_hidden = hidden_hidden
        self.bias = bias_
        self.adaptive = adaptive
        if head_num > 1: # multihead as original work
            assert hidden % head_num == 0
            self.multi_in = int(hidden/head_num)
        else: # Single head
            head_num = 1
            self.multi_in = int(hidden/head_num)

        self.head_num = head_num
        self.AFNO_1 = nn.Parameter(torch.zeros(2, head_num, self.multi_in, hidden_hidden))
        trunc_normal_(self.AFNO_1, mean = 0., std = 0.04, a = -0.2, b = 0.2)
        self.AFNO_2 = nn.Parameter(torch.zeros(2, head_num, hidden_hidden, self.multi_in))
        trunc_normal_(self.AFNO_2, mean = 0., std = 0.04, a = -0.2, b = 0.2)
        if bias_:
            self.AFNO_bias_1 = nn.Parameter(torch.zeros(2, head_num, hidden_hidden))
            self.AFNO_bias_2 = nn.Parameter(torch.zeros(2, head_num, self.multi_in))

        self.non_linearity = nn.GELU()
        # self.non_linearity = nn.LeakyReLU(0.2)
    def forward(self, x, n1 = None, n2 = None, n3 = None, n4 = None):
        # n1 and n2 are null arguments
        B, F_, d_ = x.shape
        x_multi = x.view(B, F_, self.head_num, int(d_/self.head_num))

        freq_real_1 = complex_mul(x_multi.real, self.AFNO_1[0]) - complex_mul(x_multi.imag, self.AFNO_1[1])
        freq_real_1 = self.non_linearity(freq_real_1 + self.AFNO_bias_1[0]) if self.bias else self.non_linearity(freq_real_1)
        freq_imag_1 = complex_mul(x_multi.imag, self.AFNO_1[0]) + complex_mul(x_multi.real, self.AFNO_1[1])
        freq_imag_1 = self.non_linearity(freq_imag_1 + self.AFNO_bias_1[1]) if self.bias else self.non_linearity(freq_imag_1)

        freq_real_2 = complex_mul(freq_real_1, self.AFNO_2[0]) - complex_mul(freq_imag_1, self.AFNO_2[1])
        freq_real_2 = freq_real_2 + self.AFNO_bias_2[0] if self.bias else freq_real_2
        freq_imag_2 = complex_mul(freq_imag_1, self.AFNO_2[0]) + complex_mul(freq_real_1, self.AFNO_2[1])
        freq_imag_2 = freq_imag_2 + self.AFNO_bias_2[1] if self.bias else freq_imag_2
        freq_real_2 = freq_real_2.reshape(B,F_,-1)
        freq_imag_2 = freq_imag_2.reshape(B,F_,-1)

        if not self.adaptive:
            return F_combine(freq_real_2, freq_imag_2), None, None
        else:
            f_ = F_combine(freq_real_2, freq_imag_2)
            return x * f_, f_.detach(), None

class AFF(nn.Module):
    '''
    "Adaptive Frequency Filters As Efficient Global Token Mixers"
    '''
    def __init__(self, in_dim = 32, hidden_out = 128,
                 head_num = 4, bias_ = False):
        super(AFF, self).__init__()
        self.hidden_out = hidden_out
        self.bias = bias_
        if head_num > 1: # multihead as original work
            assert in_dim % head_num == 0
            self.multi_in = int(in_dim/head_num)
        else: # Single head
            head_num = 1
            self.multi_in = int(in_dim/head_num)

        self.head_num = head_num
        self.AFNO_1 = nn.Parameter(torch.zeros(2, head_num, self.multi_in, hidden_out))
        trunc_normal_(self.AFNO_1, mean = 0., std = 0.04, a = -0.2, b = 0.2)
        self.AFNO_2 = nn.Parameter(torch.zeros(2, head_num, hidden_out, self.multi_in))
        trunc_normal_(self.AFNO_2, mean = 0., std = 0.04, a = -0.2, b = 0.2)
        if bias_:
            self.AFNO_bias_1 = nn.Parameter(torch.zeros(2, head_num, hidden_out))
            self.AFNO_bias_2 = nn.Parameter(torch.zeros(2, head_num, self.multi_in))

        self.non_linearity = nn.GELU()
        # self.non_linearity = nn.LeakyReLU(0.2)
    def forward(self, x, n1= None, n2 = None, n3 = None, n4 = None):
        B, F_, d_ = x.shape
        x_multi = x.view(B, F_, self.head_num, int(d_/self.head_num))

        freq_real_1 = complex_mul(x_multi.real, self.AFNO_1[0]) - complex_mul(x_multi.imag, self.AFNO_1[1])
        freq_real_1 = self.non_linearity(freq_real_1 + self.AFNO_bias_1[0]) if self.bias else self.non_linearity(freq_real_1)
        freq_imag_1 = complex_mul(x_multi.imag, self.AFNO_1[0]) + complex_mul(x_multi.real, self.AFNO_1[1])
        freq_imag_1 = self.non_linearity(freq_imag_1 + self.AFNO_bias_1[1]) if self.bias else self.non_linearity(freq_imag_1)

        freq_real_2 = complex_mul(freq_real_1, self.AFNO_2[0]) - complex_mul(freq_imag_1, self.AFNO_2[1])
        freq_real_2 = freq_real_2 + self.AFNO_bias_2[0] if self.bias else freq_real_2
        freq_imag_2 = complex_mul(freq_imag_1, self.AFNO_2[0]) + complex_mul(freq_real_1, self.AFNO_2[1])
        freq_imag_2 = freq_imag_2 + self.AFNO_bias_2[1] if self.bias else freq_imag_2
        freq_real_2 = freq_real_2.reshape(B,F_,-1)
        freq_imag_2 = freq_imag_2.reshape(B,F_,-1)
        filter_coeff = F_combine(freq_real_2, freq_imag_2)
        # filter_coeff = torch.stack([freq_real_2, freq_imag_2], dim=-1)
        # filter_coeff = F.softshrink(filter_coeff, lambd=0.01)
        # filter_coeff = torch.view_as_complex(filter_coeff).squeeze(-1)
        return x * filter_coeff, filter_coeff.detach(), None
    # def init_(self, init_std = 0.02, init_mean = 0.):
    #     trunc_normal_(self.AFNO_1, mean = 0, std = 0.04, a = -0.2, b = 0.2)
    #     trunc_normal_(self.AFNO_2, mean = 0, std = 0.04, a = -0.2, b = 0.2)
        # init.orthogonal_(self.Hyper_AFNO_1)
        # init.orthogonal_(self.Hyper_AFNO_2)
        # init.uniform_(self.Hyper_AFNO_1, -1, 1)
        # init.uniform_(self.Hyper_AFNO_2, -1, 1)

class INFF(nn.Module):
    '''
    The frequency filter is parameterized by SIREN 
    Constructing an implicit neural representation of frequency filter
    '''
    def __init__(self, vars: HyperVariables, 
                 adaptive = True,
                 time_ = False,
                 dropout = 0.2):
        super(INFF, self).__init__()
        self.hypervar_INFF = vars
        self.sparsity_threshold = 0.01
        self.adaptive = adaptive
        self.spec_filter = None
        self.phi_INFF = Siren_block(vars = vars,
                                hidden_dim = self.hypervar_INFF.INFF_siren_hidden,
                                out_dim = self.hypervar_INFF.hidden_dim, # [self.outdim_["in_feature"] * self.outdim_["out_feature"]]  [vars.hidden_dim, vars.hidden_dim * hidden_factor]
                                omega = self.hypervar_INFF.INFF_siren_omega,
                                siren_dim_in = self.hypervar_INFF.LFT_siren_dim_in,

                                midlayer_num = 1,
                                type_ = "linear",
                                default_init=False) # False

        self.norm_out = NormalizationLayer(norm = self.hypervar_INFF.norm_inff, # LayerNorm_seq  InstanceNorm  if not self.mixing else "None"
                                        hidden = vars.hidden_dim, 
                                        affine = False,
                                        var=True)
        self.inff_scale_bias = lft_scale_bias(self.hypervar_INFF,
                                              scale = True,
                                              bias = False,
                                              std_ = 1.0)
        # self.norm_out2 = NormalizationLayer(norm = "InstanceNorm", # LayerNorm_feature
        #                                         hidden = self.hypervar_INFF.hidden_dim, 
        #                                         affine = True)
        if time_:
            self.spec_filter = PositionwiseFeedForward(vars.hidden_dim * 2, vars.hidden_dim, 
                                                 dropout=vars.dropout, 
                                                 activation = "GeLU",
                                                 type_ = "linear-c",
                                                 layer_id= 1,
                                                 out_dim= vars.hidden_dim)
        else:
            if self.adaptive:
                self.spec_filter = CV_projection(dim_in = vars.hidden_dim, dim_out = vars.hidden_dim, # //
                                         factor= 3, non_linearity= True, hidden = None,
                                            shared = True, bias = False, spec_out= None, dropout= dropout,
                                            std_= self.hypervar_INFF.init_std, nl_type="GeLU") #self.hypervar_INFF.init_std
            
                # self.spec_filter = CV_projection_lin(dim_in = vars.hidden_dim*2, dim_out = vars.hidden_dim,
                #                                     bias = False, factor = 2,
                #                                     non_linearity = True, dropout = 0.15)
                                         
    def forward(self, x, x_cond, time_ = False):
        B, F_, d_ = x.shape
        if time_:
            f_x = self.phi_INFF(B=1, L=self.hypervar_INFF.L_base,dev = x.device)
            if self.hypervar_INFF.freq_span < self.hypervar_INFF.f_base:
                f_x = f_x[:,:self.hypervar_INFF.L_span,:]
            f_x = torch.cat((f_x.squeeze(0).expand(B,-1,-1),x_cond), dim = -1) ## 
            f_x = self.norm_out(self.spec_filter(f_x))
            f = self.hypervar_INFF.DFT_(f_x)
            x[:,1:,:] *= f[:,1:,:]
            return x, f, self.hypervar_INFF.IDFT_(f, L = self.hypervar_INFF.L_span).detach()
        else:
            f_x = self.norm_out(self.phi_INFF(B=1, L=self.hypervar_INFF.L_span,dev = x.device))
            f = self.hypervar_INFF.DFT_(f_x)
            if self.hypervar_INFF.freq_span < self.hypervar_INFF.f_base:
                f = f[:,:self.hypervar_INFF.freq_span,:]
            _, F_, d_ = f.shape
            assert x.shape[1] == f.shape[1]
            f = self.inff_scale_bias(f)
            if self.adaptive:
                x_cond = x_cond[:,:self.hypervar_INFF.freq_span,:]         
                f = f#.expand(B,-1,-1)
                # f = torch.cat((f,x_cond), dim = -1) ## 
                f = f + x_cond#.detach()
                # f = self.comp_normalization_(f)
                real_, imag_ = self.spec_filter(f)
                f = F_combine(real_, imag_)
                x[:,0:,:] *= f[:,0:,:]
            else: x = x * f
        return x, f, self.hypervar_INFF.IDFT_(f, L = self.hypervar_INFF.L_span).detach()
    def comp_normalization_(self, f):
        return f / f.norm(dim=[1,2], keepdim = True)

    def forward_2(self, x, x_cond, time_ = True):
        B, F_, d_ = x.shape
        f_x = self.phi_INFF(B=1, L=self.hypervar_INFF.L_base,dev = x.device)
        if self.hypervar_INFF.freq_span < self.hypervar_INFF.f_base:
            f_x = f_x[:,:self.hypervar_INFF.L_span,:]
        f_x = torch.cat((f_x.squeeze(0).expand(B,-1,-1),x_cond), dim = -1) ## 
        f_x = self.norm_out(self.spec_filter(f_x))
        f = self.hypervar_INFF.DFT_(f_x)
        x[:,1:,:] *= f[:,1:,:]
        return x, f, self.hypervar_INFF.IDFT_(f, L = self.hypervar_INFF.L_span).detach()


class INFF_ada(nn.Module):
    '''
    The frequency filter is parameterized by SIREN 
    Constructing an implicit neural representation of frequency filter
    '''
    def __init__(self, vars: HyperVariables, 
                 adaptive = True,
                 time_ = False,
                 dropout = 0.2):
        super(INFF_ada, self).__init__()
        self.hypervar_INFF = vars
        self.sparsity_threshold = 0.01
        self.adaptive = adaptive
        self.spec_filter = None
        self.phi_INFF = Siren_block(vars = vars,
                                hidden_dim = self.hypervar_INFF.INFF_siren_hidden,
                                out_dim = self.hypervar_INFF.hidden_dim, # [self.outdim_["in_feature"] * self.outdim_["out_feature"]]  [vars.hidden_dim, vars.hidden_dim * hidden_factor]
                                omega = self.hypervar_INFF.INFF_siren_omega,
                                siren_dim_in = self.hypervar_INFF.LFT_siren_dim_in,

                                midlayer_num = 2,
                                type_ = "linear",
                                default_init=False,
                                nl = "mix") # False

        self.norm_out = NormalizationLayer(norm = "LayerNorm_seq", # LayerNorm_seq  InstanceNorm  if not self.mixing else "None"
                                        hidden = vars.hidden_dim , 
                                        affine = True,
                                        var=True,
                                        adaptive=False,
                                        learnable_weights= False)
        self.norm_out2 = NormalizationLayer(norm = "LayerNorm_seq", # LayerNorm_seq  InstanceNorm  if not self.mixing else "None"
                                        hidden = vars.hidden_dim, 
                                        affine = False,
                                        var=True,
                                        adaptive=False,
                                        learnable_weights= False)
        self.inff_scale_bias = lft_scale_bias(self.hypervar_INFF,
                                              scale = True,
                                              bias = True,
                                              std_ = 0.5)
        # self.norm_out2 = NormalizationLayer(norm = "InstanceNorm", # LayerNorm_feature
        #                                         hidden = self.hypervar_INFF.hidden_dim, 
        #                                         affine = True)
        if time_:
            # self.spec_filter = PositionwiseFeedForward(vars.hidden_dim, vars.hidden_dim * 2, 
            #                                      dropout=vars.dropout, 
            #                                      activation = "ReLU2",
            #                                      type_ = "linear",
            #                                      layer_id= 1,
            #                                      out_dim= vars.hidden_dim,
            #                                      std= self.hypervar_INFF.init_std,
            #                                      init_ = 2)
            self.spec_filter = CV_projection(dim_in = vars.hidden_dim, dim_out = vars.hidden_dim, # //
                                         factor= 1, non_linearity= True, hidden = None,
                                            shared = True, bias = False, spec_out= None, dropout= dropout,
                                            std_= self.hypervar_INFF.init_std, nl_type="ReLU") # GeLU 
            
            # self.spec_filter = AFNO(vars.hidden_dim,
            #                 hidden_hidden=vars.hidden_dim * 1,
            #                 head_num= 1,
            #                 bias_ = False) # GeLU 
            
        else:
            if self.adaptive:
                self.spec_filter = CV_projection(dim_in = vars.hidden_dim * 2, dim_out = vars.hidden_dim, # //
                                         factor= 2, non_linearity= True, hidden = None,
                                            shared = True, bias = False, spec_out= None, dropout= dropout,
                                            std_= self.hypervar_INFF.init_std, nl_type="ReLU") #self.hypervar_INFF.init_std
                
                # self.pre_spec_filter = CV_projection(dim_in = vars.hidden_dim, dim_out = vars.hidden_dim, # //
                #                          factor= 1, non_linearity= True, hidden = None,
                #                             shared = True, bias = False, spec_out= None, dropout= dropout,
                #                             std_= self.hypervar_INFF.init_std, nl_type="ReLU") #self.hypervar_INFF.init_std  ReLU
                pass
        # self.f_x = nn.Parameter(torch.randn(1,self.hypervar_INFF.L_span,vars.hidden_dim))
    def forward(self, x, x_cond = None, x_cond2 = None, time_ = False, temporal_loc = None):
        B, F_, d_ = x.shape
        # f_x = self.phi_INFF(B=B, L=self.hypervar_INFF.L_span,dev = x.device, x = x_cond)
        if time_:
            # f_x = self.norm_out(self.phi_INFF(B=1, L=self.hypervar_INFF.L_span,dev = x.device).squeeze(0).expand(B,-1,-1), x_cond)
            # f_x = torch.cat((f_x,x_cond), dim = -1)

            f_x = self.phi_INFF(tc = temporal_loc, B=1, L=self.hypervar_INFF.L_span,dev = x.device)#.squeeze(0).expand(B,-1,-1)
            # f_x = self.f_x
            # f_x = self.norm_out(torch.cat((f_x,x_cond.detach()), dim = -1), cond = None)

            # f_x = torch.cat((self.norm_out(f_x, None), self.norm_out2(x_cond, None).detach()), dim = -1) # PRETTY GOOD

            f_x = (self.norm_out(f_x, None) + self.norm_out2(x_cond, None).detach())

            # f_x = self.norm_out(f_x + x_cond.detach())
            f = self.hypervar_INFF.DFT_(f_x)
            f = self.spec_filter(f)
            # f = F_combine(real_, imag_)
        else:
            f_x = self.norm_out(self.phi_INFF(B=1, L=self.hypervar_INFF.L_span,dev = x.device).squeeze(0).expand(B,-1,-1), cond = x_cond2.detach())
            f = self.hypervar_INFF.DFT_(f_x)
        if self.hypervar_INFF.freq_span < self.hypervar_INFF.f_base:
            f = f[:,:self.hypervar_INFF.freq_span,:]
        _, F_, d_ = f.shape
        assert x.shape[1] == f.shape[1]
        f = self.inff_scale_bias(f)

        if self.adaptive and not time_:
            # x_cond = x_cond[:,:self.hypervar_INFF.freq_span,:]    .detach()     

            f = torch.cat((f,self.hypervar_INFF.DFT_(x_cond)), dim = -1) ## 

            # cond_r, cond_i = self.pre_spec_filter(self.hypervar_INFF.DFT_(x_cond))
            # f = f + self.hypervar_INFF.DFT_(x_cond)
            # f = self.comp_normalization_(f)
            
            f = self.spec_filter(f)
            # f = F_combine(real_, imag_)
            pass
        # x[:,1:,:] *= f[:,1:,:]
        x *=f
        return x, f, f_x# self.hypervar_INFF.IDFT_(f, L = self.hypervar_INFF.L_span).detach()
    def comp_normalization_(self, f):
        return f / f.norm(dim=[1,2], keepdim = True)

    def forward_2(self, x, x_cond, time_ = True):
        B, F_, d_ = x.shape
        f_x = self.phi_INFF(B=1, L=self.hypervar_INFF.L_base,dev = x.device)
        if self.hypervar_INFF.freq_span < self.hypervar_INFF.f_base:
            f_x = f_x[:,:self.hypervar_INFF.L_span,:]
        f_x = torch.cat((f_x.squeeze(0).expand(B,-1,-1),x_cond), dim = -1) ## 
        f_x = self.norm_out(self.spec_filter(f_x))
        f = self.hypervar_INFF.DFT_(f_x)
        x[:,1:,:] *= f[:,1:,:]
        return x, f, self.hypervar_INFF.IDFT_(f, L = self.hypervar_INFF.L_span).detach()

class ForuierChannelMixer(nn.Module):
    def __init__(self, var: HyperVariables, hidden = 32, factor = 3,
                 bias = False, shared = False):
        super(ForuierChannelMixer, self).__init__()
        self.hyper_var_FCM = var
        self.hidden = hidden
        self.half_hidden = (hidden // 2 + 1)
        self.factor = factor
        self.bias = bias
        self.shared = shared

        self.FCM_1 = nn.Parameter(torch.zeros(1 if self.shared else 2, self.half_hidden, self.half_hidden * self.factor))
        trunc_normal_(self.FCM_1, mean = 0., std = 0.02, a = -2.0, b = 2.0)
        self.FCM_2 = nn.Parameter(torch.zeros(1 if self.shared else 2, self.half_hidden * self.factor, self.half_hidden))
        trunc_normal_(self.FCM_2, mean = 0., std = 0.02, a = -2.0, b = 2.0)
        if bias:
            self.FCM_bias_1 = nn.Parameter(torch.zeros(1 if self.shared else 2, 1, self.half_hidden * self.factor))
            self.FCM_bias_2 = nn.Parameter(torch.zeros(1 if self.shared else 2, 1, self.half_hidden))

        self.non_linearity = nn.GELU()
    def forward(self, x, n1 = None, n2 = None):
        # n1 and n2 are null arguments
        B, F_, d_ = x.shape
        x_c_freq = self.hyper_var_FCM.DFT_(x, dim = -1) 

        freq_real_1 = complex_mul(x_c_freq.real, self.FCM_1[0]) - complex_mul(x_c_freq.imag, self.FCM_1[0 if self.shared else 1])
        freq_real_1 = self.non_linearity(freq_real_1 + self.FCM_bias_1[0]) if self.bias else self.non_linearity(freq_real_1)
        freq_imag_1 = complex_mul(x_c_freq.imag, self.FCM_1[0]) + complex_mul(x_c_freq.real, self.FCM_1[0 if self.shared else 1])
        freq_imag_1 = self.non_linearity(freq_imag_1 + self.FCM_bias_1[0 if self.shared else 1]) if self.bias else self.non_linearity(freq_imag_1)

        freq_real_2 = complex_mul(freq_real_1, self.FCM_2[0]) - complex_mul(freq_imag_1, self.FCM_2[0 if self.shared else 1])
        freq_real_2 = freq_real_2 + self.FCM_bias_2[0] if self.bias else freq_real_2
        freq_imag_2 = complex_mul(freq_imag_1, self.FCM_2[0]) + complex_mul(freq_real_1, self.FCM_2[0 if self.shared else 1])
        freq_imag_2 = freq_imag_2 + self.FCM_bias_2[0 if self.shared else 1] if self.bias else freq_imag_2
        freq_real_2 = freq_real_2.reshape(B,F_,-1)
        freq_imag_2 = freq_imag_2.reshape(B,F_,-1)
        return self.hyper_var_FCM.IDFT_(F_combine(freq_real_2, freq_imag_2), L = self.hidden, dim = -1)
    

class ForuierChannelMixer2(nn.Module):
    def __init__(self, var: HyperVariables, hidden = 32, factor = 3,
                 bias = False, shared = False, dropout = 0.2, std_ = None,
                 nl = True):
        super(ForuierChannelMixer2, self).__init__()
        self.hyper_var_FCM = var
        self.hidden = hidden
        self.half_hidden = (hidden // 2 + 1)
        self.factor = factor
        self.bias = bias
        self.shared = shared

        self.FCM = CV_projection(dim_in = self.half_hidden, dim_out = self.half_hidden, # //
                                         factor= factor, non_linearity= nl, hidden = None,
                                            shared = self.shared, bias = bias, spec_out= None,
                                            dropout=dropout, std_= self.hyper_var_FCM.init_std if std_ is None else std_)

    def forward(self, x, n1 = None, n2 = None):
        # n1 and n2 are null arguments
        B, F_, d_ = x.shape
        x_c_freq = self.hyper_var_FCM.DFT_(x, dim = -1) 
        x_real, x_imag = self.FCM(x_c_freq)
        return self.hyper_var_FCM.IDFT_(F_combine(x_real, x_imag), L = self.hidden, dim = -1)
    



class ForuierChannelMixer3(nn.Module):
    def __init__(self, var: HyperVariables, hidden = 32, factor = 3,
                 bias = False, shared = False):
        super(ForuierChannelMixer3, self).__init__()
        self.hyper_var_FCM = var
        self.hidden = hidden
        self.half_hidden = (hidden // 2 + 1)
        self.factor = factor
        self.bias = bias
        self.shared = shared

        real_ = torch.randn(1,1, self.half_hidden) * 0.02
        imag_ = torch.randn(1,1, self.half_hidden) * 0.02
        self.channel_mixing_1 = nn.Parameter(torch.complex(real_, imag_))

        real_ = torch.randn(1,1, self.half_hidden) * 0.02
        imag_ = torch.randn(1,1, self.half_hidden) * 0.02
        self.channel_mixing_2 = nn.Parameter(torch.complex(real_, imag_))

        self.nl = Sine()
    def forward(self, x, n1 = None, n2 = None):
        # n1 and n2 are null arguments
        B, F_, d_ = x.shape
        x_c_freq = self.hyper_var_FCM.DFT_(x, dim = -1) 
        x_c_freq = self.nl(x_c_freq * self.channel_mixing_1) * self.channel_mixing_2
        return self.hyper_var_FCM.IDFT_(x_c_freq, L = self.hidden, dim = -1)