import os
import torch
import h5py
import numpy as np

class UCR(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: int,
        **kwargs,
    ):
        loc_ = os.path.join(kwargs["path"], kwargs["mod"])
        X, y = self.load_data(loc_, partition)
        super(UCR, self).__init__(X, y)
    
    # @staticmethod
    def load_data(self, data_loc, partition):
        if partition == "train":
            type_ = '_TRAIN.h5'
            X = torch.tensor(np.array(h5py.File(data_loc+type_,'r')['data']), dtype=torch.float32)
            y = torch.tensor(np.array(h5py.File(data_loc+type_,'r')['label']), dtype=torch.float32) 
            y = self.rearrange_y(y)

        elif partition == "val":
            X = {}
            y = {}
        elif partition == "test":
            type_ = '_TEST.h5'
            X = torch.tensor(np.array(h5py.File(data_loc+type_,'r')['data']), dtype=torch.float32)
            y = torch.tensor(np.array(h5py.File(data_loc+type_,'r')['label']), dtype=torch.float32) 
            y = self.rearrange_y(y)
        else:
            raise NotImplementedError("the set {} is not implemented.".format(set))
        return X, y.to(torch.int64)
    def rearrange_y(self,y):
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
            return y.reshape(n)#, number_of_class # shift so the 
        else: # need to modify 
            if min_ == -1 and number_of_class == 2:
                if max_ == 1:
                    y = torch.where(y == -1, torch.tensor(0.), y)
            return y.reshape(n)#, number_of_class