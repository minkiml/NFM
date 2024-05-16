import torch.nn as nn
from utils.vars_ import HyperVariables
from model.norms import NormalizationLayer

class Classifier(nn.Module):
    '''
    Classification head
    '''
    def __init__(self, 
                 vars: HyperVariables,
                 factor_reduce = 1,
                 pooling_method = "AVG",
                 processing ="TD"):
        super(Classifier, self).__init__() 
        self.hyper_vars_fore = vars
        final_hidden_dim = self.hyper_vars_fore.hidden_dim
        self.processing = processing
        self.factor_reduce = factor_reduce
        if processing =="TD":
            factor_reduce = factor_reduce if self.hyper_vars_fore.freq_span == self.hyper_vars_fore.f_base else self.hyper_vars_fore.freq_span
            self.pooling_ = nn.AdaptiveAvgPool1d(factor_reduce) if pooling_method == "AVG" else nn.AdaptiveMaxPool1d(factor_reduce)
            self.classifier = nn.Linear(int(final_hidden_dim * factor_reduce), self.hyper_vars_fore.class_num, bias = True)
        elif processing == "mean":
            self.norm = NormalizationLayer(norm = "LayerNorm", 
                                hidden = self.hyper_vars_fore.hidden_dim, 
                                affine = False)
            self.classifier = nn.Linear(self.hyper_vars_fore.hidden_dim, self.hyper_vars_fore.class_num, bias = True)

        self.weights_initialization()
    def forward(self, x):
        B, L_base, h = x.shape
        if self.processing =="TD":
            x = x.permute(0, 2, 1) # B, h, L
            x = self.pooling_(x)
            x = x.reshape(B,-1)
            logit = self.classifier(x)
            return logit
        elif self.processing == "mean":
            x = self.norm(x).mean(1) # B, h
            logit = self.classifier(x)
            return logit

    def nn_initialization(self, m):
        if isinstance(m, (nn.Linear)):
            pass
            if m.bias is not None:
                pass
            if m.bias is not None:
                pass
    def weights_initialization(self):
        self.init_mean = 0.
        self.init_std = self.hyper_vars_fore.init_std
        for _, m in self.named_children():
            m.apply(self.nn_initialization)
