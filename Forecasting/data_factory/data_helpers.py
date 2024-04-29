import torch
import numpy as np

def helper_y(y):
    '''
    Format class labels y into integers from 0 to k - 1 and counts the number of class (k) 
    '''
    n = y.shape[0]
    y = y.reshape(-1)
    
    # check 
    number_of_class = torch.unique(y, return_counts=True)[0].size(0)
    min_ = y.min()
    max_ = y.max()
    
    if max_ - min_ + 1 == number_of_class:
        y= y - min_
        return y.reshape(n, 1), number_of_class # shift so the 
    else: # need to modify 
        if min_ == -1 and number_of_class == 2:
            if max_ == 1:
                y = torch.where(y == -1, torch.tensor(0.), y)
        return y.reshape(n, 1), number_of_class
    
def apply_look_back_window(x,
                          L=10, 
                          S=2,

                          horizons_ = 96,
                          target = False):
    all_L, C = x.shape
    if target == False:
        M = (all_L - L) // S + 1  # Calculate the number of windows
        out_ = np.zeros((M, L, C), dtype=x.dtype)
        for j in range(M):
            start = j * S
            end = start + L
            out_[j, :, :] = x[start:end, :]
        return out_
    else:
        M = ((all_L-horizons_) - L) // S + 1  # Calculate the number of windows, letting S = 1
        out_ = np.zeros((M, L, C), dtype=x.dtype)
        out_y = np.zeros((M, horizons_ , C), dtype=x.dtype)
        for j in range(M):
            start = j * S
            end = start + L
            out_[j, :, :] = x[start:end, :]
            out_y[j, :, :] = x[end:end+horizons_, :]
        return torch.tensor(out_,dtype=torch.float32), torch.tensor(out_y,dtype=torch.float32)