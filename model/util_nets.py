import math
import torch
import torch.nn as nn
import numpy as np
from model.utilities import (trunc_normal_, F_combine, clones, Permute)
from utils.vars_ import HyperVariables

def Linear1d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    A point-wise convolution based linear layer.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


class CV_MLP(nn.Module):
    '''
    Complex-valued MLP
    '''
    def __init__(self, dim_in, dim_out, shared = False, bias = True, factor = 1,
                 non_linearity = True,  nl_type = "GeLU", n_slope = 0.01): 
        super(CV_MLP, self).__init__()     
        self.f1 = dim_in
        self.f2 = dim_out * factor
        self.shared = shared
        self.bias = bias
        self.non_linearity = non_linearity
        factor = 1 if not self.non_linearity else factor
        Linear = nn.Linear
        # TODO: this needs tidying up
        if self.shared:
            self.shared_linear = Linear(dim_in, dim_out * factor, bias = False)                
        else:
            self.real_linear = Linear(dim_in, dim_out * factor, bias = False)
            self.imag_linear = Linear(dim_in, dim_out * factor, bias = False)
        if bias:
            self.projection_bias = nn.Parameter(torch.zeros(2, 1, 1, dim_out* factor))
        if non_linearity:
            if nl_type == "Identity":
                self.nl = nn.Identity() 
            elif nl_type == "ReLU":
                self.nl = nn.ReLU()
            elif nl_type == "GeLU":
                self.nl = nn.GELU()
            elif nl_type == "LeakyReLU":
                self.nl = nn.LeakyReLU(n_slope, inplace = True)
            if self.shared:
                self.shared_linear2 = Linear(dim_out * factor, dim_out, bias = False)   
            else:
                self.real_linear2 = Linear(dim_out * factor, dim_out, bias = False)
                self.imag_linear2 = Linear(dim_out * factor, dim_out, bias = False)
            if bias:
                self.projection_bias2 = nn.Parameter(torch.zeros(2, 1, 1, dim_out))

    def forward(self, x):
        dim_ = x.dim()
        if self.shared:
            B = x.shape[0]
            x_real_imag = torch.cat((x.real,x.imag), dim = 0)
            x_real_imag = self.shared_linear(x_real_imag)
            real_part = x_real_imag[:B] - x_real_imag[B:]
            imag_part = x_real_imag[B:] + x_real_imag[:B]
        else:
            real_part = self.real_linear(x.real) - self.imag_linear(x.imag)
            imag_part = self.real_linear(x.imag) + self.imag_linear(x.real)
        if self.bias:
            real_part = real_part + self.projection_bias[0] if dim_ == 3 else real_part + self.projection_bias[0].squeeze(0)
            imag_part = imag_part + self.projection_bias[1] if dim_ == 3 else imag_part + self.projection_bias[1].squeeze(0)

        if self.non_linearity:
            real_part = self.nl(real_part)
            imag_part = self.nl(imag_part)
            if self.shared:
                x_real_imag = torch.cat((real_part,imag_part), dim = 0)
                x_real_imag = self.shared_linear2(x_real_imag)
                real_part2 = x_real_imag[:B] - x_real_imag[B:]
                imag_part2 = x_real_imag[B:] + x_real_imag[:B]
            else:
                real_part2 = self.real_linear2(real_part) - self.imag_linear2(imag_part)
                imag_part2 = self.real_linear2(imag_part) + self.imag_linear2(real_part)
            if self.bias:
                real_part2 = real_part2 + self.projection_bias2[0] if dim_ == 3 else real_part2 + self.projection_bias2[0].squeeze(0)
                imag_part2 = imag_part2 + self.projection_bias2[1] if dim_ == 3 else imag_part2 + self.projection_bias2[1].squeeze(0)
            return F_combine(real_part2, imag_part2)
        else: return F_combine(real_part, imag_part)

class PositionwiseFeedForward(nn.Module):
    "FFN"
    def __init__(self, d_model, d_ff, dropout=0.1, activation = "GeLU", n_slope = 0.2,
                 type_ = "linear", layer_id = 1, out_dim = None, std = None, bias = False,
                 init_ = 1):
        super(PositionwiseFeedForward, self).__init__()
        self.type_ = type_
        self.d_model = d_model
        self.d_ff = d_ff
        self.std = std
        Linear = nn.Linear if type_ == "linear" else Linear1d
        self.w_1 = Linear(d_model, d_ff, bias = bias)
        self.w_2 = Linear(d_ff, out_dim if out_dim is not None else d_model, bias = bias) ## 
       
        self.dropout = nn.Dropout(dropout)
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(n_slope, inplace = True)
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GeLU":
            self.activation = nn.GELU()
        elif activation == "Sine":
            self.activation = Sine()
        else:
            self.activation = nn.Identity()
        
        self.weight_initalization(layer_id = layer_id, init_=init_)
    def forward(self, x):
        if self.type_ != "linear":
            x = x.permute(0,2,1)
        x = self.dropout(self.w_2(self.dropout(self.activation(self.w_1(x)))) ) 
        if self.type_ != "linear":
            x = x.permute(0,2,1)
        return x
    def nn_initialization(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                pass
            if m.bias is not None:
                nn.init.zeros_(m.bias)  
    def weight_initalization(self,layer_id = 1, init_ = 1):        
        if init_ == 1:
            for var, m in self.named_children():
                if var == "w_1":
                    self.div = self.d_model
                    trunc_normal_(m.weight, std=0.02 if self.std is None else self.std)
                    m.apply(self.nn_initialization)
                elif var == "w_2":
                    self.div = self.d_ff
                    trunc_normal_(m.weight, std=0.02 if self.std is None else self.std)
                    m.weight.data.div_(math.sqrt(2.0 * layer_id)) 
                    m.apply(self.nn_initialization)
        elif init_ == 2:
            for var, m in self.named_children():
                m.apply(self.nn_initialization)

class Periodic_activation(nn.Module):
    def __init__(self, nl = "sin"):
        super(Periodic_activation, self).__init__()
        self.nl = nl
    def forward(self, x):
        if self.nl == "sin":
            return torch.sin(x)
        elif self.nl == "cos":
            return torch.cos(x)
        elif self.nl == "mix":
            B, L, C = x.shape
            x_sined = torch.sin(x[:,:,::2])
            x_cosined = torch.cos(x[:,:,1::2])
            return torch.stack((x_sined,x_cosined), dim=3).view(B, L, -1)

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
    def forward(self, input):
        return torch.sin(input)
    
class mul_omega(nn.Module):
    def __init__(self, 
                 feature_dim,
                 omega,
                 omega_learnable = False):
        super(mul_omega, self).__init__() 
        self.omega = nn.Parameter(torch.rand((1,1,feature_dim)) * omega) if omega_learnable else omega

    def forward(self, x):
        return x * self.omega

'''
SIREN ("Implicit Neural Representations with Periodic Activation Functions"
https://github.com/vsitzmann/siren/tree/master) 
+
Fourier Features ("Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")
'''
class Siren_block(nn.Module):
    def __init__(self, 
                    vars: HyperVariables,
                    hidden_dim,
                    out_dim,
                    omega,
                    siren_dim_in = None,

                    midlayer_num = 1,
                    type_ = "linear",
                    default_init = False,
                    nl = "sin"): 
        super(Siren_block, self).__init__() 
        self.hyper_var_siren = vars
        self.default_init = default_init
        midlayer_num = midlayer_num
        self.omega = omega
        self.dim_in = siren_dim_in
        self.hidden_dim = hidden_dim
        if self.hyper_var_siren.tau_in_inrs == "independent":
            self.FF_mapping = fourier_mapping(ff_dim = siren_dim_in,  
                                                ff_sigma=self.hyper_var_siren.ff_std, # !!
                                                learnable_ff = True, 
                                                ff_type = "gaussian", # deterministic_exp  gaussian
                                                L = self.hyper_var_siren.L_base)
        Linear = nn.Linear if type_ == "linear" else Linear1d
        self.phi_init = nn.Sequential(Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                        Linear(siren_dim_in,hidden_dim, bias = False),
                                        Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                        mul_omega(hidden_dim, omega, False),
                                        Periodic_activation(nl=nl))
        
        self.phi_mid = clones(nn.Sequential(Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                            Linear(hidden_dim,hidden_dim, bias = False),
                                            Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                            mul_omega(hidden_dim, omega, False),
                                            Periodic_activation(nl=nl)), midlayer_num)

        self.phi_last = nn.Sequential(Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                      Linear(hidden_dim, out_dim, bias = False),
                                      Permute((0,2,1)) if type_ != "linear" else nn.Identity(),)
 
        self.siren_initialization()
    def forward(self, tc = None, L = None, 
        dev = None):
        t_len = self.hyper_var_siren.L_base if L is None else L
        if tc is None:
            t = self.FF_mapping(L=t_len, dev = dev)
        else: t = tc
        t = self.phi_init(t)
        for midlayer in (self.phi_mid):
            t = midlayer(t)
        t = self.phi_last(t)
        return t
    
    def init_firstLayer(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                if not self.default_init:
                    m.weight.data.uniform_(-1 / self.dim_in, 1 / self.dim_in)
            if m.bias is not None:
                m.bias.data.zero_() 
        else: pass
    def init_midLayers(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                if not self.default_init:
                    m.weight.data.uniform_(
                        -np.sqrt(6.0 / self.dim_in) / self.omega,
                        np.sqrt(6.0 / self.dim_in) / self.omega)
            if m.bias is not None:
                m.bias.data.zero_()   
        else: pass
    def init_lastLayer(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                if not self.default_init:
                    m.weight.data.uniform_(
                        -np.sqrt(6.0 / self.dim_in) / (self.omega),
                        np.sqrt(6.0 / self.dim_in) / (self.omega))
            if m.bias is not None:
                m.bias.data.zero_()                    
    def siren_initialization(self):
        for var, m in self.named_children():
            if var == "phi_init": 
                m.apply(self.init_firstLayer)
            elif var == "phi_mid":
                self.dim_in = self.hidden_dim
                m.apply(self.init_midLayers)
            elif var == "phi_last": 
                self.dim_in = self.hidden_dim
                m.apply(self.init_lastLayer)

class fourier_mapping(nn.Module):
    def __init__(self, ff_dim,  
                 ff_sigma=512,
                 learnable_ff = False,
                 ff_type = "gaussian",
                 L = 512):
        super(fourier_mapping, self).__init__()      
        assert (ff_dim % 2) == 0
        
        # ff 
        self.ff_dim_half = int(ff_dim / 2)
        self.ff_sigma = ff_sigma
        input_dim = 1 # 1d modality (sequence)
        self.ff_type = ff_type
        if ff_type == "deterministic": # nerf style
            ff_linear = 2 ** torch.linspace(0, self.ff_sigma, self.ff_dim_half // input_dim)
        elif ff_type == "deterministic_exp":
            log_freqs = torch.linspace(0, np.log(self.ff_sigma), self.ff_dim_half // input_dim)
            ff_linear = torch.exp(log_freqs)
        elif ff_type == "gaussian":
            ff_linear = torch.randn(input_dim, self.ff_dim_half) * self.ff_sigma
        self.ff_linear = nn.Parameter(ff_linear, requires_grad=learnable_ff)

        ## coord sampler
        self.coord_sampler = CoordSampler(L = L)
    def forward(self, L = None, dev = None):
        coord = self.coord_sampler(L = L, device = dev)
        if self.ff_type == "deterministic" or self.ff_type == "deterministic_exp":
            fourier_features = torch.matmul(coord, self.ff_linear.unsqueeze(0))
            fourier_features = fourier_features.view(1, coord.shape[1], -1)
        elif self.ff_type == "gaussian":
            fourier_features = torch.matmul(coord, self.ff_linear)
        if not self.ff_type == "deterministic":
            fourier_features = fourier_features * np.pi
        fourier_features = [torch.cos(fourier_features), torch.sin(fourier_features)]
        fourier_features = torch.cat(fourier_features, dim=-1)
        return fourier_features

def shape2coordinate(spatial_shape, batch_size, min_value=-1.0, max_value=1.0, upsample_ratio=1, device=None):
    coords = []
    for num_s in spatial_shape:
        num_s = int(num_s * upsample_ratio)
        _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
        _coords = min_value + (max_value - min_value) * _coords
        coords.append(_coords)
    coords = torch.meshgrid(*coords, indexing="ij")
    coords = torch.stack(coords, dim=-1)
    ones_like_shape = (1,) * coords.ndim
    coords = coords.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    return coords

class CoordSampler(nn.Module):
    """Generates coordinate inputs for time series (coord dim = 1).
    """
    def __init__(self, L):
        super().__init__()
        self.coord_range = [-1,1]
        self.L = L
    def base_sampler(self, L = None, coord_range=None, upsample_ratio=1.0, device=None):
        coord_range = self.coord_range if coord_range is None else coord_range
        min_value, max_value = coord_range
        batch_size = 1
        spatial_shape = [self.L] if L is None else [L]

        return shape2coordinate(spatial_shape, batch_size, min_value, max_value, upsample_ratio, device)

    def forward(self, L = None, coord_range=None, upsample_ratio=1.0, device=None):
        coords = self.base_sampler(L, coord_range, upsample_ratio, device)
        return coords # (1, Length, 1)