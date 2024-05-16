import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import math
import torch.fft as fft
import math, copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])    

def F_combine(real, imag, dim_ = -1):
    freq_= torch.stack((real, imag), dim = dim_)
    return torch.view_as_complex(freq_) # F, d

class Squeeze(nn.Module):
    def __init__(self, dim_ = 3):
        super(Squeeze, self).__init__()
        self.dim_ = dim_
    def forward(self, x):
        return x.squeeze(self.dim_)

class Permute(nn.Module):
    def __init__(self, permutation_order):
        super(Permute, self).__init__()
        self.permutation_order = permutation_order

    def forward(self, x):
        return x.permute(*self.permutation_order)
    
class Flatten(nn.Module):
    def __init__(self, start_dim):
        super(Flatten, self).__init__()
        self.start_dim = start_dim

    def forward(self, x):
        return torch.flatten(x, start_dim = 1, end_dim= -1)

class View(nn.Module):
    def __init__(self, view_to_):
        super(View, self).__init__()
        self.view_to_ = view_to_

    def forward(self, x):
        return x.view(*self.view_to_)

class NoiseInjection(nn.Module):
    def __init__(self, scale = 0.01, 
                 hidden_dim = 32,
                 fixed = False):
        super(NoiseInjection, self).__init__()
        self.scale = scale
        self.fixed = fixed
        if fixed:
            self.n = torch.randn((1, 1, hidden_dim), requires_grad=False) * self.scale
    def forward(self, x):
        if self.fixed:
            return x + self.n.to(x.device)
        else:
            return x + (torch.randn((x.shape), requires_grad=False).to(x.device) * self.scale)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def drop_path(x, keep_prob: float = 1.0, inplace: bool = False) :
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
    mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
    mask.div_(keep_prob)
    if inplace:
        x.mul_(mask)
    else:
        x = x * mask
    return x

class DropPath(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if self.training and self.p > 0:
            x = drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"