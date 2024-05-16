import torch
import os
import logging
import pandas as pd

from sklearn.preprocessing import StandardScaler
from Forecasting.data_factory.data_macro import *
from torch.utils.data import Dataset, DataLoader
from Forecasting.data_factory.data_helpers import (apply_look_back_window)

class forecasting_dataset(Dataset):
    def __init__(self, 
            set_ = [],
            data_name = "",
            type_ = "training"):
        super(forecasting_dataset, self).__init__()

        # Basic info
        self.name = data_name
        self.type_ = type_

        self.train_x = None
        self.train_y = None
        self.train_freq_target = None
        self.set_dataset(set_[0], set_[1], set_[2])
    def set_dataset(self, x, y, z_ = None):
        self.train_x = x
        self.train_y = y
        self.train_freq_target = z_
    def __len__(self):
        return self.train_x.shape[0] 
    def __getitem__(self, index):            
        return self.train_x[index], self.train_y[index], \
            self.train_freq_target[index] if self.train_freq_target is not None else 1
    
class Forcasting_data(object):
    def __init__(
                # Loaded dataset
                self, path = "", 
                sub_dataset = "",
                varset_train = [], # [Fs, F_in, L_out (horizon), L_in (lookback)]
                varset_test = [],
                channel_independence = False,
                # Forecasting setup
                training_portion= 0.7, 
                look_back= 96,
                horizon = 96,
                
                # loader params
                batch_training = 32,
                batch_testing = 32,
                num_workers = 0):
        # Global logger
        log_loc = os.environ.get("log_loc")
        root_dir = os.path.abspath(os.path.join(os.getcwd(), '..')) # go to one root above
        logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all'), level=logging.INFO,
                            format = '%(asctime)s - %(name)s - %(message)s')
        self.logger = logging.getLogger('From forecasting_dataset')


        self.path = path
        self.sub_dataset = sub_dataset
        self.training_portion = training_portion
        self.channel_independence = channel_independence

        self.look_back = look_back
        self.horizon = horizon

        self.varset_train = varset_train
        self.varset_test = varset_test
        # loader params
        self.batch_ = batch_training
        self.batch_testing = batch_testing
        self.num_workers = num_workers
        self.__read_and_construct__()

    def __read_and_construct__(self):
        list_of_benchmark_protocol = LIST_OF_FORCASTING_BENCHMARK_LENGTH            
        path_ = self.path #root_dir
        type_ = '.csv'

        self.scaler = StandardScaler() # note that when calling testing data, use the same setup for normalization 
        raw_pd = pd.read_csv(os.path.join(path_, self.sub_dataset + type_))
        raw_pd = raw_pd[raw_pd.columns[1:]] # for all cont datasets, the first column is date info

        # Training/val/testing setting, following the conventional way
        total_length = len(raw_pd)
        if self.sub_dataset == 'ETTh1' or self.sub_dataset == 'ETTh2':
            train_p = list_of_benchmark_protocol[0]["training_length"]
            val_p = [list_of_benchmark_protocol[0]["val_length"][0] - self.look_back, list_of_benchmark_protocol[0]["val_length"][1]]
            test_p = [list_of_benchmark_protocol[0]["testing_length"][0] - self.look_back, list_of_benchmark_protocol[0]["testing_length"][1]]
        elif self.sub_dataset == 'ETTm1' or self.sub_dataset == 'ETTm2':
            train_p = list_of_benchmark_protocol[1]["training_length"]
            val_p = [list_of_benchmark_protocol[1]["val_length"][0] - self.look_back, list_of_benchmark_protocol[1]["val_length"][1]]
            test_p = [list_of_benchmark_protocol[1]["testing_length"][0] - self.look_back, list_of_benchmark_protocol[1]["testing_length"][1]]
        else:
            num_train = int(total_length * self.training_portion)
            num_test = int(total_length * 0.2)
            num_vali = total_length - num_train - num_test
            train_p = [0, num_train]
            val_p = [num_train - self.look_back, num_train + num_vali]
            test_p = [total_length - num_test - self.look_back,total_length]

        self.logger.info(f"(training) time step from: {train_p[0]} to {train_p[1]}")
        self.logger.info(f"(val) time step from: {val_p[0]} to {val_p[1]}")
        self.logger.info(f"(testing) time step from: {test_p[0]} to {test_p[1]}")

        training_data = raw_pd[train_p[0]: train_p[1]]

        # z-score normalization
        self.scaler.fit(training_data.values)
        raw_pd = self.scaler.transform(raw_pd.values)


        training_data = raw_pd[train_p[0]: train_p[1]] # (training length, C)
        validation_data = raw_pd[val_p[0]: val_p[1]]
        testing_data = raw_pd[test_p[0]: test_p[1]]
        self.C_ = training_data.shape[1]
        self.L_ = self.look_back

        # Lookback window 
        train_x, train_y = apply_look_back_window(training_data, 
                                            L =self.look_back,
                                            S = 1,
                                            horizons_ = self.horizon,
                                            target = True) 
        testing_x, testing_y = apply_look_back_window(testing_data, 
                                            L =self.look_back,
                                            S = 1,
                                            horizons_ = self.horizon,
                                            target = True) 
        
        f_in_factor =  int(self.varset_train[1] / self.varset_test[1])
        testing_x = testing_x[:,::f_in_factor,:]

        f_out_factor =  int(self.varset_train[0] / self.varset_test[0])
        testing_y = testing_y[:,::f_out_factor,:]

        val_x, val_y = apply_look_back_window(validation_data, 
                                            L =self.look_back,
                                            S = 1,
                                            horizons_ = self.horizon,
                                            target = True) 
        if self.channel_independence:
            Bx, Lx, Cx = train_x.shape
            By, Ly, Cy = train_y.shape
            train_x, train_y = train_x.permute(0,2,1).reshape(-1,1,Lx).permute(0,2,1), train_y.permute(0,2,1).reshape(-1,1,Ly).permute(0,2,1)

        self.logger.info(f"Channel independence: {self.channel_independence}")
        self.logger.info(f"number of train x samples: {train_x.shape[0]}")
        self.logger.info(f"number of val x samples: {val_x.shape[0]}")
        self.logger.info(f"number of test x samples: {testing_x.shape[0]}")

        train_freq_target = torch.fft.rfft(torch.cat((train_x, train_y), dim = 1), dim = 1, norm = "ortho") 

        # Construct dataloaders
        self.training_loader = DataLoader(forecasting_dataset(set_ = [train_x, train_y, train_freq_target],
                                            data_name = self.sub_dataset,
                                            type_ = "training"),
                                        batch_size = self.batch_,
                                        shuffle = True,
                                        num_workers= self.num_workers,
                                        drop_last=True)
        self.logger.info(f"Training data loader is constructed. The total number of mini-batches: {len(self.training_loader)}")
        self.testing_loader = DataLoader(forecasting_dataset(set_ = [testing_x, testing_y, None],
                                            data_name = self.sub_dataset,
                                            type_ = "testing"),
                                        batch_size = self.batch_testing,
                                        shuffle = False,
                                        num_workers= self.num_workers,
                                        drop_last=False)
        self.logger.info(f"Testing data loader is constructed. The total number of mini-batches: {len(self.testing_loader)}")
        self.val_loader = DataLoader(forecasting_dataset(set_ = [val_x, val_y, None],
                                            data_name = self.sub_dataset,
                                            type_ = "validating"),
                                        batch_size = self.batch_testing,
                                        shuffle = False,
                                        num_workers= self.num_workers,
                                        drop_last=False)
        self.logger.info(f"Validation data loader is constructed. The total number of mini-batches: {len(self.val_loader)}")

        del val_x, val_y, testing_x, testing_y, train_x, train_y, train_freq_target
    def __get_training_loader__(self):
        return self.training_loader
    def __get_testing_loader__(self):
        return self.testing_loader
    def __get_val_loader__(self):
        return self.val_loader
    
    def __get_info__(self):
        return (self.L_, self.C_)
    def __inverse_normalize__(self, x):
        # X must be original multivariate feature shape
        if self.C_ != 1:
            assert x.shape[-1] != 1

        if x.dim() == 3:
            B, L, C = x.shape
            x = x.reshape(-1,C)
            x = self.scaler.inverse_transform(x.detach().numpy())
            x = x.reshape(B, L, C)
        elif x.dim() == 2:
            L, C = x.shape
            x = self.scaler.inverse_transform(x.detach().numpy())
        return x
