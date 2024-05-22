import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from utils.vars_ import HyperVariables
from model.util_nets import Siren_block
from model.norms import NormalizationLayer
from model.utilities import F_combine

class LFT_block(nn.Module):
    "Learnable frequency token embeddings"
    def __init__(self, 
                 vars:HyperVariables):
        super(LFT_block, self).__init__() 
        self.hyper_vars_LFT = vars
        self.lft = vars.lft
        if self.lft:
            self.FT_generator = Siren_block(vars,
                                            hidden_dim = self.hyper_vars_LFT.LFT_siren_hidden,
                                            out_dim = self.hyper_vars_LFT.hidden_dim,
                                            omega = self.hyper_vars_LFT.LFT_siren_omega,
                                            siren_dim_in = self.hyper_vars_LFT.LFT_siren_dim_in,
                                            
                                            midlayer_num = 2,
                                            type_ = "linear",
                                            default_init= False,
                                            nl = "mix")
            self.norm_out = NormalizationLayer(norm = "InstanceNorm", 
                                hidden = self.hyper_vars_LFT.hidden_dim, 
                                affine = False)
            
            if self.hyper_vars_LFT.LFT_nomalization: 
                self.norm_out2 = NormalizationLayer(norm = "InstanceNorm", 
                                hidden = self.hyper_vars_LFT.hidden_dim, 
                                affine = False)
            else:
                self.norm_out2 = nn.Identity()
        
        else:
            # This should not be accounted for in the model size
            self.zero_base = nn.Parameter(torch.zeros(1,self.hyper_vars_LFT.freq_span,
                                                       self.hyper_vars_LFT.LFT_siren_hidden, dtype=torch.complex64), 
                                                       requires_grad=False)
            # self.naive = nn.Parameter(torch.randn(1,self.hyper_vars_LFT.freq_span, self.hyper_vars_LFT.LFT_siren_hidden, dtype=torch.complex64) * 0.1)

        self.lft_scale_bias = lft_scale_bias(self.hyper_vars_LFT,
                                       scale=True, ## 
                                       bias=True, ## 
                                       std_ =1.0)
        
    def forward(self, x, temporal_loc = None):
        B, f_in, hidden = x.shape
        x_freq = self.hyper_vars_LFT.DFT_(self.norm_out2(x))
        # Mapping Input freq to the frequency base      
        f_tokens = self.generate_frequency_token(B=B, dev = x_freq.device, temporal_loc = temporal_loc)
        if self.lft:
            f_tokens = self.lft_scale_bias(f_tokens)        

        # x_base = self.hyper_vars_LFT.map_to_base(x_freq * self.hyper_vars_LFT.scale_factor * self.hyper_vars_LFT.scale_factor_IN_to_OUT, f_tokens)
        x_base = self.hyper_vars_LFT.map_to_base2(x_freq * self.hyper_vars_LFT.scale_factor * self.hyper_vars_LFT.scale_factor_IN_to_OUT, f_tokens)
        x_base = self.hyper_vars_LFT.IDFT_(x_base, L = self.hyper_vars_LFT.L_span) 
        
        if not self.lft:
            x_base = x_base
        return x_base

    def generate_frequency_token(self, B, dev = None, temporal_loc = None):
        if self.lft:
            base_signal = self.norm_out(self.FT_generator(tc = temporal_loc, 
                                                          L = self.hyper_vars_LFT.L_span, 
                                                          dev = dev).squeeze(0).expand(B,-1,-1))
            f_tokens = self.hyper_vars_LFT.DFT_(base_signal)
        else:
            base_signal = None
            f_tokens = self.zero_base.expand(B,-1,-1)

        if self.hyper_vars_LFT.freq_span < self.hyper_vars_LFT.f_base:
            f_tokens = f_tokens[:,:self.hyper_vars_LFT.freq_span,:]
        return f_tokens

class lft_scale_bias(nn.Module):
    ''' 
    Simple learnable Fourier basis
    
    '''
    def __init__(self, 
                 vars: HyperVariables,
                 scale = False,
                 bias = False,
                 std_ = 0.1):
        super(lft_scale_bias, self).__init__() 
        self.hyper_var_LFB = vars
        self.hidden_dim = self.hyper_var_LFB.hidden_dim
        self.scale = scale
        self.std_ = std_
        if scale:
            self.lft_scale_real = nn.Parameter(torch.ones(1,1,self.hidden_dim) * self.std_) # DC & AC
            self.lft_scale_imag = nn.Parameter(torch.ones(1,1,self.hidden_dim) * self.std_)
            
        self.bias = bias
        if bias:
            self.lft_bias_real = nn.Parameter(torch.zeros(1,1,self.hidden_dim)) # DC & AC
            self.lft_bias_imag = nn.Parameter(torch.zeros(1,1,self.hidden_dim))
        self.init_()
    def forward(self, lft):
        if self.scale:
            lft = lft * F_combine(self.lft_scale_real, self.lft_scale_imag, -1)
        if self.bias:
            lft = lft + F_combine(self.lft_bias_real, self.lft_bias_imag) 
        return lft
    
    def init_(self):
        pass

class PositionalEncoding(nn.Module):
    '''
    Sine-cos positional encoding used in Vanilla Transformer
    '''
    def __init__(self, d_model,  max_len=720, scaler = False):
        super(PositionalEncoding, self).__init__()        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.max_len = max_len

        if scaler:
            self.scaler = nn.Parameter(torch.ones(1,1,d_model))
        else: self.scaler = 1.
    def forward(self, B):
        # pe = self.pe
        pe = Variable(self.pe[:, :], 
                         requires_grad=False)
        return pe.expand(B,-1,-1) * self.scaler