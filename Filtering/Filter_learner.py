import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import time

from Filtering.NFM_ import model_constructor
from utils.optimizer import opt_constructor
from Filtering.data_factory.TS_dataloader import Load_dataset
from utils.vars_ import HyperVariables
from utils.logger_ import Value_averager, Logger, grad_logger_spec
import matplotlib.pyplot as plt

from thop import profile

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

class EarlyStopping:
    def __init__(self, patience=7, verbose=False,dataset_name='', delta=0, logger = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = -np.Inf

        self.early_stop = False
        self.delta = delta
        self.dataset=dataset_name
        self.logger = logger

    def __call__(self, acc, model, path):
        if acc > self.best_acc:
            self.save_checkpoint(acc, model, path)
            self.best_acc = acc
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(f'No validation improvement & EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, acc, model, path):
        if self.verbose:
            self.logger.info(f'Validation improved (MSE: {self.best_acc:.6f} --> {acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, '_checkpoint.pth'))


class Solver(object):
    DEFAULTS = {}
    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        log_loc = os.environ.get("log_loc")
        root_dir = os.getcwd() 
        logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all.txt'), level=logging.INFO,
                            format = '%(asctime)s - %(name)s - %(message)s')
        self.logger = logging.getLogger('Solver')
        seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        data, (_, _) = Load_dataset({"data_path": self.data_path,
                                    "sub_dataset": self.dataset,
                                    "varset_train": self.vars_in_train,

                                    "max_modes": self.max_modes,
                                    "diversity": self.diversity,
                                    "num_modes": self.num_modes,
                                    
                                    "filter_type": self.filter_mode,
                                    "class_num_filter": self.num_class,

                                    "batch_training": self.batch,
                                    "batch_testing": self.batch_testing})
        
        self.data_to_fit = data.__get_training_loader__()
        del data

        self.loss_CE = Value_averager()

        self.device = torch.device(f'cuda:{self.gpu_dev}' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"GPU (device: {self.device}) used" if torch.cuda.is_available() else 'cpu used')

        self.build_model()

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.plots_save_path):
            os.makedirs(self.plots_save_path)
        if not os.path.exists(self.his_save_path):
            os.makedirs(self.his_save_path)
        self.log = Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            ('%.5f', 'loss_CE'),
                            ('%.5f', 'lr'),
                            ('%.5f', 'wd'),
                            ('%.4e', 'forecaster_'), 
                            ('%.4e', 'projection_out'), 
                            ('%.4e', 'projection_in'),

                            ('%.4e', 'phi_INFF'),
                            ('%.4e', 'spec_filter'),
                            ('%.4e', 'LFT'),
                            ('%.4e', 'TD_filter_NFF'),
                            ('%.4e', 'nff_fl'),
                            # ('%.4e', 'projection_in')
                            )
        
    def build_model(self):
        # self.model = Model(DotDict({'seq_len': self.win_size//self.DSR, 'enc_in': self.input_c, 'individual': self.individual,'cut_freq':self.cutfreq,'pred_len':self.win_size-self.win_size//self.DSR}))
        self.hyper_variables = HyperVariables(sets_in_training = self.vars_in_train,
                                            sets_in_testing = self.vars_in_test,
                                            freq_span = self.freq_span,
                                            C_ = self.input_c,
                                            channel_dependence = self.channel_dependence,
                                            dropout = self.dropout, 

                                            filter_type = self.filter_type,
                                            hidden_dim = self.hidden_dim,
                                            hidden_factor = self.hidden_factor,
                                            inff_siren_hidden = self.inff_siren_hidden,
                                            inff_siren_omega = self.inff_siren_omega,
                                            layer_num = self.layer_num,

                                            lft_siren_dim_in = self.siren_in_dim,
                                            lft_siren_hidden = self.siren_hidden,
                                            lft_siren_omega = self.siren_omega,
                                            
                                            loss_type= self.loss_type,
                                            class_num = self.num_class)
        self.model = model_constructor(self.hyper_variables)

        ipe = len(self.data_to_fit)
        self.optimizer, self.ir_scheduler, self.wd_scheduler = opt_constructor(self.scheduler,
                                                                            self.model,
                                                                            lr = self.lr_,

                                                                            warm_up = int(self.n_epochs* ipe * self.warm_up),
                                                                            fianl_step = int(self.n_epochs* ipe),
                                                                            start_lr = self.start_lr,
                                                                            ref_lr = self.ref_lr,
                                                                            final_lr = self.final_lr,
                                                                            start_wd = self.start_wd,
                                                                            final_wd = self.final_wd)

        if torch.cuda.is_available():
            self.model.to(self.device)
    def fit(self):
        self.logger.info("======================Fitting MODE======================")
        train_steps = len(self.data_to_fit)
        self.logger.info(f'train_steps: {train_steps}')
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.n_epochs):
            speed_t = []
            epoch_time = time.time()
            self.model.train()
            self.hyper_variables.training_set()
            for i, (x, y, _, _, _) in enumerate(self.data_to_fit):
                input = x.to(self.device)######################
                y = y.to(self.device)
                if self.ir_scheduler is not None and  self.wd_scheduler is not None:
                    _new_lr = self.ir_scheduler.step()
                    _new_wd = self.wd_scheduler.step()
                self.optimizer.zero_grad()

                per_itr_time = time.time()
                y_pred, z, inff, s, res, t_token, f_token, zz, freq2  = self.model(input)
                CE_loss = self.model.criteria_cla(logit = y_pred, target = y.detach())

                loss = CE_loss
                loss.backward()
                self.optimizer.step()
                speed_t.append(time.time() - per_itr_time)

                self.loss_CE.update(CE_loss.item())
                grad_stats_AC = grad_logger_spec(self.model.named_parameters(), prob_ = "forecaster_") # // # 'LFTLayer_1'  learnable_freq_base_real_AC
                grad_stats_conv_IMAG = grad_logger_spec(self.model.named_parameters(), prob_ = "projection_out") # 'LFTLayers'  learnable_freq_base_imag
                grad_stats_exp_real = grad_logger_spec(self.model.named_parameters(), prob_ = "projection_in")

                grad_stats_1 = grad_logger_spec(self.model.named_parameters(), prob_ = "phi_INFF") # 'LFTLayers'  learnable_freq_base_imag
                grad_stats_2 = grad_logger_spec(self.model.named_parameters(), prob_ = "spec_filter")
                grad_stats_22 = grad_logger_spec(self.model.named_parameters(), prob_ = "FT_generator")
                grad_stats_3 = grad_logger_spec(self.model.named_parameters(), prob_ = "TD_filter_NFF") # 'LFTLayers'  learnable_freq_base_imag
                grad_stats_4 = grad_logger_spec(self.model.named_parameters(), prob_ = "nff_fl")

                self.log.log_into_csv_(epoch+1,
                                            i,
                                            self.loss_CE.avg,
                                            _new_lr if self.ir_scheduler is not None else 0.,
                                            _new_wd if self.wd_scheduler is not None else 0.,
                                            grad_stats_AC.avg,
                                            grad_stats_conv_IMAG.avg,
                                            grad_stats_exp_real.avg,
                                            
                                            grad_stats_1.avg,
                                            grad_stats_2.avg,
                                            grad_stats_22.avg,
                                            grad_stats_3.avg,
                                            grad_stats_4.avg)
            self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}] & speed per epoch: {(time.time() - epoch_time): .5f}, Loss_CE:{self.loss_CE.avg: .4f}")
            self.logger.info("")  
            self.logger.info("******Visualization*******")  
            if ((epoch+1) % 50) == 0:
                self.log.log_2d_vis(self.hyper_variables.IDFT_(inff)[0,:,:], inff[0,:,:], f"{epoch+1}_nff")
                self.log.frequency_reponse(inff, range = -1, name_ = f"{epoch+1}_nff")
                self.log.log_2d_vis(z[0,:,:], self.hyper_variables.DFT_(z)[0,:,:], f"{epoch+1}_out_NFF")
                self.log.log_2d_vis(s, self.hyper_variables.DFT_(s), f"{epoch+1}_timefilter")
                self.log.log_2d_vis(res, self.hyper_variables.DFT_(res), f"{epoch+1}_res")
                self.log.log_2d_vis(t_token, f_token, f"{epoch+1}_LFT")
                self.log.log_2d_vis(zz[0,:,:], self.hyper_variables.DFT_(zz)[0,:,:], f"{epoch+1}_preout_NFF")
                self.log.log_2d_vis(self.hyper_variables.IDFT_(freq2)[0,:,:] , freq2[0,:,:] , f"{epoch+1}_INFF_out")
                self.log.log_2d_vis(x[0,:,:], self.hyper_variables.DFT_(x)[0,:,:], f"{epoch+1}_input")
                self.logger.info("")  
           