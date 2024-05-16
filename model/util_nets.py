import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.init as init
import numpy as np
from model.norms import NormalizationLayer
from model.utilities import (trunc_normal_, F_combine, clones, Permute)
from utils.vars_ import HyperVariables
class CV_projection(nn.Module):
    '''
    Complex-valued vector projection
    '''
    def __init__(self, dim_in, dim_out, shared = False, bias = True, factor = 1, spec_out = None,
                 non_linearity = False, hidden=None, dropout = 0.2, std_ = 0.02,
                 nl_type = "GeLU"): 
        super(CV_projection, self).__init__()     
        self.f1 = dim_in
        self.f2 = dim_out * factor
        self.shared = shared
        self.bias = bias
        self.non_linearity = non_linearity
        self.std_ = std_
        factor = 1 if not self.non_linearity else factor
        Linear = nn.Linear
        if self.shared:
            self.shared_linear = Linear(dim_in, dim_out * factor if hidden is None else hidden, bias = False)                
        else:
            self.real_linear = Linear(dim_in, dim_out * factor if hidden is None else hidden, bias = False)
            self.imag_linear = Linear(dim_in, dim_out * factor if hidden is None else hidden, bias = False)
        if bias:
            self.projection_bias = nn.Parameter(torch.zeros(2, 1, 1, dim_out* factor if hidden is None else hidden))
        if non_linearity:
            if nl_type == "Identity":
                self.nl = nn.Identity() # nn.ReLU() #nn.LeakyReLU(0.2) # ModReLU(False)
            elif nl_type == "ReLU":
                self.nl = nn.ReLU()
            elif nl_type == "GeLU":
                self.nl = nn.GELU()
            elif nl_type == "LeakyReLU":
                self.nl = nn.LeakyReLU(0.2)
            if self.shared:
                self.shared_linear2 = Linear(dim_out * factor if hidden is None else hidden, spec_out if spec_out is not None else dim_out, bias = False)   
            else:
                self.real_linear2 = Linear(dim_out * factor if hidden is None else hidden, spec_out if spec_out is not None else dim_out, bias = False)
                self.imag_linear2 = Linear(dim_out * factor if hidden is None else hidden, spec_out if spec_out is not None else dim_out, bias = False)
            if bias:
                self.projection_bias2 = nn.Parameter(torch.zeros(2, 1, 1, spec_out if spec_out is not None else dim_out))
        # self.dropout = nn.Dropout(dropout)
        self.weight_initalization()
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
            # comp = self.nl(F_combine(real_part, imag_part))
            # real_part, imag_part = comp.real, comp.imag
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
    def nn_initialization(self, m):
        if isinstance(m, nn.Linear):
            if m.weight is not None:
                if self.std_ != 1.:    
                    # m.weight.data.uniform_(
                    #         -np.sqrt(1.0 / self.factor_norm),
                    #         np.sqrt(1.0 / self.factor_norm))
                    # init.normal_(m.weight, mean = 0., std =  0.02)
                    # init.xavier_uniform_(m.weight)
                    # init.uniform_(m.weight, mean = self.init_mean, std =  0.02)

                    # trunc_normal_(m.weight if isinstance(m, nn.Linear) else m, 0., 0.02, -0.2, 0.2)
                    
                    # init.kaiming_uniform_(m.weight, mode="fan_out")

                    # init.xavier_uniform_(m.weight, gain = 1.4)

                    # nn.init.orthogonal_(m.weight, gain = 1)
                    
                    # init.xavier_normal_(m.weight, gain=0.02)

                    #m.weight.data.normal_(0, 0.02) * 0.5
                    # init.kaiming_normal_(m.weight, a= 0.2, mode = "fan_in", nonlinearity='leaky_relu')
                #     init.orthogonal_(m.weight)
                    pass
            if m.bias is not None:
                print("conv_bias")
                nn.init.zeros_(m.bias)   
    def weight_initalization(self):
        for var, m in self.named_children():
            if var == "shared_linear" or var == "real_linear" or var == "imag_linear": 
                self.factor_norm = self.f1
            elif var == "shared_linear2" or var == "real_linear2" or var == "imag_linear2":
                self.factor_norm = self.f2         
            m.apply(self.nn_initialization)
def Linear1d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    A Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
class CV_projection_lin(nn.Module):
    '''
    Complex-valued vector projection
    # this is equivalent to CV linears with the weights for real and image being shared (as above) 
    '''
    def __init__(self, dim_in, dim_out, bias = False, factor = 1,
                 non_linearity = False, dropout = 0.): 
        super(CV_projection_lin, self).__init__()     
        self.non_linearity = non_linearity
        hidden = dim_out if not non_linearity else dim_out*factor

        self.cv_linear1 = nn.Linear(dim_in, hidden, bias = bias).to(torch.cfloat)
        self.nl = nn.ReLU() if non_linearity else nn.Identity()
        self.cv_linear2 = nn.Linear(hidden, dim_out, bias = bias).to(torch.cfloat) if non_linearity else nn.Identity()
    def forward(self, x):
        x = self.cv_linear1(x)
        x = F_combine(self.nl(x.real), self.nl(x.imag))
        return self.cv_linear2(x)
    
class PositionwiseFeedForward(nn.Module):
    "FFN"
    def __init__(self, d_model, d_ff, dropout=0.1, activation = "GeLU",
                 type_ = "linear", layer_id = 1, out_dim = None, std = None, bias = False,
                 no_init = False, init_ = 1, num_layers = 2):
        super(PositionwiseFeedForward, self).__init__()
        self.type_ = type_
        self.d_model = d_model
        self.d_ff = d_ff
        self.std = std
        self.num_layers = num_layers
        Linear = nn.Linear if type_ == "linear" else Linear1d
        self.w_1 = Linear(d_model, d_ff, bias = bias)
        if num_layers == 3:
            self.w_2 = Linear(d_ff, d_ff, bias = bias) ## 
            self.w_3 = Linear(d_ff, out_dim if out_dim is not None else d_model, bias = bias) ## 
        elif num_layers == 2:
            self.w_2 = Linear(d_ff, out_dim if out_dim is not None else d_model, bias = bias) ## 
        else:
            self.w_1 = Linear(d_model, out_dim if out_dim is not None else d_model, bias = bias)
        self.dropout = nn.Dropout(dropout)
        if activation == "ReLU":
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == "ReLU2":
            self.activation = nn.ReLU()
        elif activation == "GeLU":
            self.activation = nn.GELU()
        elif activation == "Sine":
            self.activation = Sine()
        else:
            self.activation = nn.Identity()
        if no_init:
            pass
        else: self.weight_initalization(init_=init_)
    def forward(self, x):
        if self.type_ != "linear":
            x = x.permute(0,2,1)
        if self.num_layers == 3:
            x = self.dropout(self.w_3(self.activation(self.w_2(self.dropout(self.activation(self.w_1(x)))))))
        elif self.num_layers == 2:
            x = self.dropout(self.w_2(self.dropout(self.activation(self.w_1(x)))) ) 
        else:
            x = self.activation(self.w_1(x))

        if self.type_ != "linear":
            x = x.permute(0,2,1)
        return x
    def nn_initialization(self, m):
        if isinstance(m, (nn.Linear)):
            if m.weight is not None:
                # m.weight.data.uniform_(
                #         -np.sqrt(1.0 / self.div),
                #         np.sqrt(1.0 / self.div))
                pass
                # init.xavier_uniform_(m.weight, gain = 1.4)
            if m.bias is not None:
                nn.init.zeros_(m.bias)  
        if isinstance(m, (nn.Conv1d)):
            if m.weight is not None:
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                print("FFT keiming init")
            if m.bias is not None:
                nn.init.zeros_(m.bias)  
    def weight_initalization(self,layer_id = 1, init_ = 1):
        self.init_mean = 0.
        self.init_std = 0.01
        pass
        if init_ == 1:
            for var, m in self.named_children():
                if var == "w_1":
                    self.div = self.d_model
                    trunc_normal_(m.weight, std=0.02 if self.std is None else self.std)
                    m.apply(self.nn_initialization)
                    print(F"FFN std: {self.std}")
                elif var == "w_2":
                    self.div = self.d_ff
                    if init_ == 1:
                        trunc_normal_(m.weight, std=0.02 if self.std is None else self.std)
                        if self.num_layers != 3:
                            m.weight.data.div_(math.sqrt(2.0 * layer_id)) 
                        m.apply(self.nn_initialization)
                        print(F"FFN std: {self.std}")
                elif var == "w_3":
                    self.div = self.d_ff
                    trunc_normal_(m.weight, std=0.02 if self.std is None else self.std)
                    m.weight.data.div_(math.sqrt(2.0 * layer_id))
                    m.apply(self.nn_initialization)
        elif init_ == 2:
            for var, m in self.named_children():
                m.apply(self.nn_initialization)

class GLU_projection(nn.Module):
    def __init__(self, dim_in, out_dim, act = "Linear", projection_omega = 1.,
                 type_ = "linear"):
        super(GLU_projection, self).__init__()
        self.type_ = type_
        self.out_dim = out_dim
        Linear = nn.Linear if type_ == "linear" else Linear1d
        hidden_dim = out_dim * 4
        assert (hidden_dim % 2) == 0
        self.inter_dim = int(hidden_dim / 2) # !!
        self.projection_in = Linear(dim_in, hidden_dim, bias = True)
        self.projection_in_out = Linear(self.inter_dim, out_dim, bias = False)
        self.act = {'Linear': nn.Identity(),
                'periodic': Sine(),
                }[act]
        self.mul_omega = mul_omega(feature_dim = self.out_dim, 
                                omega = projection_omega)
    def forward(self, x):
        if self.type_ != "linear":
            x = x.permute(0,2,1)
        x = self.projection_in(x)
        if self.type_ != "linear":
            x = self.projection_in_out(self.periodic_act(x[:,:self.inter_dim,:] * (x[:,self.inter_dim:,:]).sigmoid()))
        else:
            x = self.projection_in_out(self.periodic_act(x[:,:,:self.inter_dim] * (x[:,:,self.inter_dim:]).sigmoid()))
        if self.type_ != "linear":
            x = x.permute(0,2,1)
        return x
    def periodic_act(self, x):
        if isinstance (self.act, (Sine)):
            if self.type_ != "linear":
                return self.act(self.mul_omega(x[:,:self.inter_dim,:]))
            else: return self.act(self.mul_omega(x[:,:,:self.inter_dim]))
        else: return x            

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, bias = False, init = 1):
        super(TokenEmbedding, self).__init__()
        padding = 1 
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=bias)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.weight is not None:
                    if init == 1:
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    elif init == 2:
                        nn.init.xavier_normal_(m.weight, gain=1.4)
                    # nn.init.kaiming_uniform_(m.weight, mode='fan_in')
                    elif init == 3:
                        pass
                    # init.xavier_uniform_(m.weight)
                    # init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()   
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# class TsEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TsEmbedding, self).__init__()
#         self.hidden = d_model
#         self.half_hidden = (d_model // 2 + 1)
#         self.tokenConv = CV_projection(dim_in = self.half_hidden, dim_out = d_model, # //
#                                          factor= 2, non_linearity= True, hidden = None,
#                                             shared = True, bias = False, spec_out= None,
#                                             dropout=dropout, std_= self.hyper_var_FCM.init_std if std_ is None else std_)
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 if m.weight is not None:
#                     # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                     nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

#                     # init.xavier_uniform_(m.weight)
#                     # init.xavier_normal_(m.weight, gain=0.02)
#                 if m.bias is not None:
#                     m.bias.data.zero_()   
#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return x

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
    def forward(self, input):
        return torch.sin(input)
    
class mul_omega(nn.Module):
    def __init__(self, 
                 feature_dim,
                 omega,
                 omega_learnable = False,
                 gaussian = False,
                 spectrum = 100,
                 max = 0):
        super(mul_omega, self).__init__() 
        if gaussian:
            rand_ = torch.randn((1,1,feature_dim)) * 0.1
        else: rand_ = torch.zeros((1,1,feature_dim))
        # self.omega = nn.Parameter((torch.randn((1,1,feature_dim)) * spectrum) * omega) if omega_learnable else omega
        self.omega = nn.Parameter((torch.rand((1,1,feature_dim))* rand_) * omega) if omega_learnable else omega

    def forward(self, x):
        return x * self.omega

class Fourier_feature_emb(nn.Module):
    '''
    '''
    def __init__(self, d_model,  
                 max_len=720,
                 omega= 1.):
        super(Fourier_feature_emb, self).__init__()        
        # Compute the positional encodings once in log space.
        self.d_model = d_model
        self.max_len = max_len # L_input or L_base used during training
        self.b_ = torch.randn(1,int(d_model/2)) * omega
        self.a_ = torch.ones_like(self.b_)

    def forward(self, B = 1, L = None, t = None):
        length_ = self.max_len if L is None else L
        pe = torch.zeros(length_, self.d_model)
        position = torch.linspace(0., 1, length_).unsqueeze(1).unsqueeze(0).repeat(B,1,1) if t is None else t
        # position += (torch.randn(B, self.max_len,1) * 1/self.max_len * 0.2)
        pe[:, 0::2] = torch.sin(torch.pi* 2 * position * self.b_.unsqueeze(0)) * self.a_.unsqueeze(0)
        pe[:, 1::2] = torch.cos(torch.pi* 2 * position * self.b_.unsqueeze(0)) * self.a_.unsqueeze(0)
        return pe.unsqueeze(0)
        
class Siren_block(nn.Module):
    '''
    Learnable frequency token based on SIREN ("Implicit Neural Representations with Periodic Activation Functions"
                                                  https://github.com/vsitzmann/siren/tree/master)
    A purpose of learnable frequency token is to model rich prior about input sequences and add it to constructed fourier spectrum   
    '''
    def __init__(self, 
                    vars: HyperVariables,
                    hidden_dim,
                    out_dim,
                    omega,
                    siren_dim_in = None,

                    midlayer_num = 1,
                    type_ = "linear",
                    default_init = False,
                    ff_mapping = "ff",
                    nl = "sin",
                    shared_loc = False): # "conv_linear"
        super(Siren_block, self).__init__() 
        self.hyper_var_siren = vars
        self.default_init = default_init
        midlayer_num = midlayer_num
        self.omega = omega
        self.dim_in = siren_dim_in
        self.ff_mapping = ff_mapping
        self.hidden_dim = hidden_dim
        if ff_mapping == "naive":
            self.FF_mapping = Fourier_feature_emb(siren_dim_in, max_len=self.hyper_var_siren.L_base, # TODO: frequency base layer not chaning
                                        omega=omega)
        elif ff_mapping == "ff":
            self.FF_mapping = fourier_mapping(ff_dim = siren_dim_in,  
                                            ff_sigma=256, # 256
                                            learnable_ff = True, # !!
                                            ff_type = "gaussian", # deterministic_exp  gaussian
                                            L = self.hyper_var_siren.L_base)
        Linear = nn.Linear if type_ == "linear" else Linear1d
        self.phi_init = nn.Sequential(Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                        Linear(siren_dim_in,hidden_dim, bias = False),
                                        Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                        NormalizationLayer(norm = "None", # self.hyper_var_siren.norm, 
                                        hidden = hidden_dim, 
                                        affine = True),
                                        mul_omega(hidden_dim, omega, False),
                                        Periodic_activation(nl=nl))
        
        self.phi_mid = clones(nn.Sequential(Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                            Linear(hidden_dim,hidden_dim, bias = False),
                                            Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                 NormalizationLayer(norm = "None",#self.hyper_var_siren.norm, 
                                                    hidden = hidden_dim, 
                                                    affine = True),
                                 mul_omega(hidden_dim, omega, False),
                                 Periodic_activation(nl=nl)), midlayer_num)

        self.phi_last = nn.Sequential(Permute((0,2,1)) if type_ != "linear" else nn.Identity(),
                                      Linear(hidden_dim, out_dim, bias = False),
                                      Permute((0,2,1)) if type_ != "linear" else nn.Identity(),)
 
        self.siren_initialization()
    def forward(self, tc = None, B = 1, L = None, 
        dev = None):
        t_len = self.hyper_var_siren.L_base if L is None else L # self.hyper_var_siren.L_span
        if tc is None:
            if self.ff_mapping == "naive":
                t = self.FF_mapping(B, L=t_len).to(dev) # (B, L_base, d_model)
            elif self.ff_mapping == "ff":
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
                    print("siren_init")
                else:
                    print("!!!")
            if m.bias is not None:
                pass
                # m.bias.data.zero_() 
        else: pass
    def init_midLayers(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                if not self.default_init:
                    m.weight.data.uniform_(
                        -np.sqrt(6.0 / self.dim_in) / self.omega,
                        np.sqrt(6.0 / self.dim_in) / self.omega)
                    print("siren_init")
                else:
                    print("!!!")
                pass
            if m.bias is not None:
                pass
                m.bias.data.zero_()   
        else: pass
    def init_lastLayer(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None:
                if not self.default_init:
                    m.weight.data.uniform_(
                        -np.sqrt(6.0 / self.dim_in) / (self.omega),
                        np.sqrt(6.0 / self.dim_in) / (self.omega))
                    print("siren_init")
                else:
                    print("!!!")
                # else:
                pass
                # m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                pass
                m.bias.data.zero_()                    
        else: pass
    def siren_initialization(self):
        for var, m in self.named_children():
            # For all initialization here, we assume n = 1 (i.e., INR function of time only)
            if var == "phi_init": 
                # m.apply(self.init_firstLayer)
                m.apply(self.init_midLayers)
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
    # def forward(self, L = None, dev = None):
    #     coord = self.coord_sampler(L = L, device = dev)
        
    #     if self.ff_type == "deterministic" or self.ff_type == "deterministic_exp":
    #         fourier_features = torch.matmul(coord.unsqueeze(-1), self.ff_linear.unsqueeze(0))
    #         fourier_features = fourier_features.view(*coord.shape[:-1], -1)
    #     elif self.ff_type == "gaussian":
    #         fourier_features = torch.matmul(coord, self.ff_linear)
    #     if not self.ff_type == "deterministic":
    #         fourier_features = fourier_features * np.pi

    #     fourier_features = [torch.cos(fourier_features), torch.sin(fourier_features)]
    #     fourier_features = torch.cat(fourier_features, dim=-1)
    #     return fourier_features

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
    
class Input_embedding(nn.Module):
    def __init__(self, in_channel,
                 hidden_dim,
                 nonlinearity = "sin",
                 group_fe = False,
                 spect = 100):
        super(Input_embedding, self).__init__()     
        self.group_fe = group_fe
        self.input_projection = nn.Linear(in_channel, hidden_dim, bias = False)

        if not group_fe:
            self.fourier_embedding = nn.Sequential(
                                                nn.Linear(in_channel, hidden_dim, 
                                        bias = False),
                                        mul_omega(feature_dim = hidden_dim, 
                                        omega = 1,
                                        omega_learnable = True,
                                        gaussian = True,
                                        spectrum = spect),
                                        Periodic_activation(nl = nonlinearity),#nn.LeakyReLU(0.2),
                                        nn.Linear(hidden_dim, hidden_dim, 
                                        bias = False)
                                        )
        else:
            self.fourier_embedding_lf = nn.Sequential(
                                                nn.Linear(in_channel, int(hidden_dim/2), 
                                        bias = False),
                                        mul_omega(feature_dim = int(hidden_dim/2), 
                                        omega = 1,
                                        omega_learnable = True,
                                        gaussian = True,
                                        spectrum = 30,
                                        max = 0),
                                        Periodic_activation(nl = nonlinearity),#nn.LeakyReLU(0.2),
                                        nn.Linear(int(hidden_dim/2), hidden_dim, 
                                        bias = False)
                                        )
            self.fourier_embedding_hf = nn.Sequential(
                                                nn.Linear(in_channel, int(hidden_dim/2), 
                                        bias = False),
                                        mul_omega(feature_dim = int(hidden_dim/2), 
                                        omega = 1,
                                        omega_learnable = True,
                                        gaussian = True,
                                        spectrum = spect,
                                        max = 60),
                                        Periodic_activation(nl = nonlinearity),#nn.LeakyReLU(0.2),
                                        nn.Linear(int(hidden_dim/2), hidden_dim, 
                                        bias = False)
                                        )
    def forward(self, x):
        x1 = self.input_projection(x)

        if not self.group_fe:
            x2 = self.fourier_embedding(x)
            return x1 + x2
        else:
            low_x2 = self.fourier_embedding_lf(x)
            high_x2 = self.fourier_embedding_hf(x)
            return x1 + low_x2 + high_x2
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
