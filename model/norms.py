import torch
import torch.nn as nn

class NormalizationLayer(nn.Module):
    "Normalization wapper"
    def __init__(self, norm, hidden, affine = True, 
                 var = True, mean = True):
        super(NormalizationLayer, self).__init__()
        # Normalization (input shape  B, L, C)
        if norm == "LayerNorm":
            self.norm_ = LayerNorm(hidden, affine = affine)
 
        elif norm == "InstanceNorm":
            self.norm_ = InstanceNorm(hidden, affine = affine, var = var, mean = mean)
     
    def forward(self, x):
        if self.norm_ is not None:
            x  = self.norm_(x)
        return x

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

    def forward(self, x):
        dim_ = x.dim()
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True)+ 1e-5)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            if dim_ > 3:
                return self.a_2_LNF.unsqueeze(0) * x + self.b_2_LNF.unsqueeze(0)
            else: return self.a_2_LNF * x + self.b_2_LNF
        else: return x

class InstanceNorm(nn.Module):
    def __init__(self, features, affine = True,
                 eps=1e-6,  var = True, mean = True):
        super(InstanceNorm, self).__init__()
        self.affine = affine
        self.features = features
        if (affine):
            self.a_2_LNS = nn.Parameter(torch.ones(1,1,features))
            self.b_2_LNS = nn.Parameter(torch.zeros(1,1,features))
        else: self.affine = False
        self.var = var
        self.mean = mean
        self.eps = eps
    
    def forward(self, x):
        dim_ = x.dim()
        mean = x.mean(1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True)+ 1e-5)
        if self.mean:
            x = (x - mean)
        if self.var:
            x = x / (std)
     
        if self.affine:
            if dim_ > 3:
                return self.a_2_LNS.unsqueeze(0) * x + self.b_2_LNS.unsqueeze(0)
            else: 
                return self.a_2_LNS * x + self.b_2_LNS
        return x