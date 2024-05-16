# Normalizations for complex numbers of Fourier space
import torch
import torch.nn as nn
# For real-valued vectors

class NormalizationLayer(nn.Module):
    "Normalization wapper"
    def __init__(self, norm, hidden, affine = True, 
                 eps=1e-6, var = True, mean = True, dc_removal = False,
                 adaptive = False, learnable_weights = False):
        super(NormalizationLayer, self).__init__()
        # Normalization (input shape  B, L, C)
        if norm == "LayerNorm_feature":
            self.norm_ = LayerNorm(hidden, affine = affine)
        elif norm == "InstanceNorm":
            # Not flexible
            self.norm_ = InstanceNorm(hidden, affine = affine)
        
        elif norm == "LayerNorm_seq":
            self.norm_ = LayerNorm_seq(hidden, affine = affine, var = var, mean = mean, adaptive = adaptive,
                                       learnable_weights = learnable_weights)
        elif norm == "BatchNorm":
            self.norm_ = nn.BatchNorm1d(hidden)
        else:
            self.norm_ = None
        if dc_removal:
            self.norm_2 = LayerNorm_seq(hidden, affine = False, var = False, mean = True, adaptive = False)
        else: self.norm_2 = nn.Identity()
    def forward(self, x, cond = None):
        if self.norm_ is not None:
            x  = self.norm_(x, cond = cond)
        return self.norm_2(x)

class LayerNorm(nn.Module):
    "Suppose the channel (hidden) dim is last dim."
    def __init__(self, features, affine = True,
                 eps=1e-6):
        super(LayerNorm, self).__init__()
        self.affine = affine
        if affine:
            self.a_2_LNF = nn.Parameter(torch.ones(1,1,features))
            self.b_2_LNF = nn.Parameter(torch.zeros(1,1,features))
        self.eps = eps

    def forward(self, x, cond = None):
        dim_ = x.dim()
        mean = x.mean(-1, keepdim=True) # over the vectors
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True)+ 1e-5)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            if dim_ > 3:
                return self.a_2_LNF.unsqueeze(0) * x + self.b_2_LNF.unsqueeze(0)
            else: return self.a_2_LNF * x + self.b_2_LNF
        else: return x

class LayerNorm_seq(nn.Module):
    "Layernorm over sequence length dim"
    def __init__(self, features, affine = True,
                 eps=1e-6,  var = True, mean = True,
                 adaptive = False, learnable_weights =False):
        super(LayerNorm_seq, self).__init__()
        self.affine = affine
        self.adaptive = adaptive
        self.learnable_weights = learnable_weights
        self.features = features
        if (affine):
            self.a_2_LNS = nn.Parameter(torch.ones(1,1,features))
            self.b_2_LNS = nn.Parameter(torch.zeros(1,1,features))
        else: self.affine = False
        if learnable_weights:
            self.cond_mean_scale = nn.Parameter(torch.ones(1,1,features))
            # self.cond_mean_tra = nn.Parameter(torch.ones(1,1,features) * 0.01)
            self.cond_std_scale = nn.Parameter(torch.ones(1,1,features))
            # self.cond_std_tra = nn.Parameter(torch.ones(1,1,features) * 0.01)

        self.var = var
        self.mean = mean
        self.eps = eps
    
    def forward(self, x, cond = None):
        dim_ = x.dim()
        if cond is not None:
            assert x.shape == x.shape 
            mean = cond.mean(1, keepdim=True)
            std = torch.sqrt(torch.var(cond, dim=1, keepdim=True)+ 1e-5)
        else:
            mean = x.mean(1, keepdim=True)
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True)+ 1e-5)
        if self.mean:
            x = (x - mean)
        if self.var:
            x = x / (std)
        if self.adaptive:
            conditional_mean = torch.mean(cond, dim = 1, keepdim=True).detach()
            conditional_std = torch.sqrt(torch.var(cond, dim=1, keepdim=True)+ 1e-5).detach()
            if self.learnable_weights:
                conditional_std = (conditional_std * self.cond_std_scale) 
                conditional_mean = (conditional_mean * self.cond_mean_scale)
            x = (x * (conditional_std)) + conditional_mean
        if self.affine:
            if dim_ > 3:
                return self.a_2_LNS.unsqueeze(0) * x + self.b_2_LNS.unsqueeze(0)
            else: 
                return self.a_2_LNS * x + self.b_2_LNS
            # x = (x - conditional_mean) / (conditional_std + self.eps)
        return x

class InstanceNorm(nn.Module):
    '''
    Suppose the input dim is (B, L, C)
    '''
    def __init__(self, features,  affine = False, 
                 eps=1e-6, adaptive = False):
        super(InstanceNorm, self).__init__()
        self.affine = affine
        self.adaptive = adaptive
        if (affine) and (not adaptive) :
            self.a_2_inst = nn.Parameter(torch.ones(1,1,features))
            self.b_2_inst = nn.Parameter(torch.zeros(1,1,features))
        else: self.affine = False
        self.eps = eps
        
    def forward(self, x, cond = None):
        if cond is not None:
            assert x.shape == x.shape 
        mean = x.mean(dim = (1,2), keepdim=True) # over the vectors
        std = x.std(dim = (1,2), keepdim=True)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            return self.a_2_inst * x + self.b_2_inst
        elif self.adaptive:
            conditional_mean = cond.mean(dim = (1,2), keepdim=True)
            conditional_std = cond.std(dim = (1,2), keepdim=True)
            x = (x * (conditional_std + self.eps)) + conditional_mean
        return x

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    

class RevStat(nn.Module):
    # Reversible Stationarization with fourier Transform (LFC)
    def __init__(self,
                 channel = 1,
                 revstat_range = [0,1], 
                 affine = False):
        super().__init__()
        # RevStat variables
        self.channel = channel
        # window l . . . . l
        self.from_ = revstat_range[0]
        self.to_ = revstat_range[1]
        self.range_ = self.to_ - self.from_ 

        # self.affine = affine
        # if affine:
        #     self.affine_scale = nn.Parameter(torch.ones(2, 1, self.range_, channel))
        #     self.affine_shift = nn.Parameter(torch.zeros(2, 1, self.range_, channel))

    def stationarization_(self, freq, affine = None):
        B, F_, C = freq.shape
        self.LFC = freq.detach().clone()
        self.LFC[:,self.to_:,:] *= 0.
        # Learnable affine transformation
        if affine is not None:
            affine_scale = affine[0]
            affine_shift = affine[1]
            self.LFC[:,self.from_:self.to_,:].real *= affine_scale[0]
            self.LFC[:,self.from_:self.to_,:].real += affine_shift[0]
            self.LFC[:,self.from_:self.to_,:].imag *= affine_scale[1] # Suppose DC is always removed
            self.LFC[:,self.from_:self.to_,:].imag += affine_shift[1]
        freq -= self.LFC
        # self.LFC *= scale_
        return freq
    
    def de_stationarization_(self, freq):
        lfc_ = self.LFC
        # mapping TODO: mapped LFC to right fourier basis according the f_base
        assert freq.shape == lfc_.shape
        if self.affine:
            raise NotImplementedError("Mapping Affine neneds to be implemented")
        freq += lfc_
        return freq

    def __get_lfc__(self):
        return self.LFC
    
    def __set_lfc__(self, lfc):
        self.LFC = lfc

    def stationarization_2(self, freq, affine = None, affine_dc = None):
        B, F_, C = freq.shape
        self.LFC = freq.detach().clone()
        if self.range_ > 0:
            self.LFC[:,1:self.from_,:] *= 0.
            self.LFC[:,self.to_:,:] *= 0.
        else:
            self.LFC[:,1:,:] *= 0.
        # Learnable affine transformation
        if affine is not None:
            affine_scale = affine[0]
            affine_shift = affine[1]
            self.LFC[:,self.from_:self.to_,:].real *= affine_scale[0]
            self.LFC[:,self.from_:self.to_,:].real += affine_shift[0]
            self.LFC[:,self.from_:self.to_,:].imag *= affine_scale[1] # Suppose DC is always removed
            self.LFC[:,self.from_:self.to_,:].imag += affine_shift[1]
        if affine_dc is not None:
            self.LFC[:,0:1,:].real *= affine_dc[0]
            self.LFC[:,0:1,:].real += affine_dc[1]
        freq -= self.LFC
        # self.LFC *= scale_
        return freq