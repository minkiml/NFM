import torch
import torch.nn as nn

from model.utilities import trunc_normal_, F_combine
from utils.vars_ import HyperVariables
from model.norms import NormalizationLayer
from model.util_nets import (Siren_block, CV_MLP)
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

class INFF(nn.Module):
    '''
    Implicit neural Fourier filter
    '''
    def __init__(self, vars: HyperVariables, 
                 dropout = 0.2):
        super(INFF, self).__init__()
        self.hypervar_INFF = vars
        self.phi_INFF = Siren_block(vars = vars,
                                hidden_dim = self.hypervar_INFF.INFF_siren_hidden,
                                out_dim = self.hypervar_INFF.hidden_dim, 
                                omega = self.hypervar_INFF.INFF_siren_omega,
                                siren_dim_in = self.hypervar_INFF.LFT_siren_dim_in,

                                midlayer_num = 2,
                                type_ = "linear",
                                default_init=False,
                                nl = "mix") 

        self.norm_out = NormalizationLayer(norm = "InstanceNorm", 
                                        hidden = vars.hidden_dim , 
                                        affine = False)
        
        self.inff_scale_bias = lft_scale_bias(self.hypervar_INFF,
                                              scale = True,
                                              bias = True,
                                              std_ = 1.0) ##

        self.cv_mlp = CV_MLP(dim_in = vars.hidden_dim, dim_out = vars.hidden_dim, bias = False,
                            factor= 1, shared = True, nl_type="ReLU")
    
    def forward(self, x, conditional = None, temporal_loc = None):
        B, F_, d_ = x.shape
        f_x = self.phi_INFF(tc = temporal_loc, L=self.hypervar_INFF.L_span,dev = x.device)
        f_x = (self.norm_out(f_x) + conditional.detach()) # 1)
        f = self.hypervar_INFF.DFT_(f_x)
        f = self.inff_scale_bias(f)
        f = self.cv_mlp(f)
     
        if self.hypervar_INFF.freq_span < self.hypervar_INFF.f_base:
            f = f[:,:self.hypervar_INFF.freq_span,:]
        _, F_, d_ = f.shape
        assert x.shape[1] == f.shape[1]

        x *=f
        return x