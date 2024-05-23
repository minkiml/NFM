import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import time

from Forecasting.NFM_FC import model_constructor
from utils.optimizer import opt_constructor
from Forecasting.data_factory.TS_dataloader import Load_dataset
from utils.vars_ import HyperVariables
from utils.logger_ import Value_averager, Logger, grad_logger_spec
from fvcore.nn import FlopCountAnalysis
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
    def __init__(self, patience=3, verbose=False,dataset_name='', delta=0, logger = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_MSE = np.Inf
        self.best_MAE = np.Inf

        self.early_stop = False
        self.delta = delta
        self.dataset=dataset_name
        self.logger = logger

    def __call__(self, mse_, mae_, model, path):
        # if mse_ < self.best_MSE:
        # if mae_ < self.best_MAE:
        #     self.save_checkpoint(mse_,mae_, model, path)
        #     self.best_MAE = mae_
        #     self.counter = 0
        if mse_ < self.best_MSE or mae_ < self.best_MAE:
            self.save_checkpoint(mse_,mae_, model, path)
            if mse_ < self.best_MSE:
                self.best_MSE = mse_ 
            if mae_ < self.best_MAE:
                self.best_MAE = mae_
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(f'No validation improvement & EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, mse_, mae_, model, path):
        if self.verbose:
            self.logger.info(f'Validation improved (MSE: {self.best_MSE:.6f} --> {mse_:.6f}) & (MAE: {self.best_MAE} --> {mae_:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, '_checkpoint.pth'))


class Solver(object):
    DEFAULTS = {}
    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        log_loc = os.environ.get("log_loc")
        root_dir = os.getcwd() 
        logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all.txt'), level=logging.INFO,
                            format = '%(asctime)s - %(name)s - %(message)s')
        self.logger = logging.getLogger('Solver')

        data, (_, _) = Load_dataset({"data_path": self.data_path,
                                    "sub_dataset": self.dataset,
                                    "varset_train": self.vars_in_train,
                                    "varset_test": self.vars_in_test,
                                    "channel_dependence": self.channel_dependence,
                                    "training_portion": 0.7,
                                    "look_back": self.look_back,
                                    "horizon": self.horizon,

                                    "batch_training": self.batch,
                                    "batch_testing": self.batch_testing})
        
        self.training_data = data.__get_training_loader__()
        self.testing_data = data.__get_testing_loader__()
        self.val_data = data.__get_val_loader__()
        del data

        self.loss_TD = Value_averager()
        self.loss_FD = Value_averager()
        self.loss_total = Value_averager()

        self.device = torch.device(f'cuda:{self.gpu_dev}' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"GPU (device: {self.device}) used" if torch.cuda.is_available() else 'cpu used')
        self.peak_memory_init = torch.cuda.max_memory_allocated(self.device)
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
                            ('%.5f', 'loss_TD'),
                            ('%.5f', 'loss_FD'),
                            ('%.5f', 'lr'),
                            ('%.5f', 'wd'),
                            ('%.4e', 'forecaster_'), 
                            ('%.4e', 'll_NFF'), 
                            ('%.4e', 'projection_in'),

                            ('%.4e', 'NFF_block'),
                            ('%.4e', 'cv_mlp'),
                            
                            ('%.4e', 'channel_mixer'),
                            ('%.4e', 'LFT'),
                            )
        
    def build_model(self):
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
                                            lft = bool(self.lft),
                                            lft_siren_dim_in = self.siren_in_dim,
                                            lft_siren_hidden = self.siren_hidden,
                                            lft_siren_omega = self.siren_omega,
                                            lft_norm= bool(self.lft_norm),
                                            tau= self.tau,
                                            
                                            loss_type= self.loss_type,
                                            norm_trick= self.norm_trick,
                                            ff_projection_ex = self.ff_projection_ex)
        self.model = model_constructor(self.hyper_variables)
        ipe = len(self.training_data)
        self.optimizer, self.lr_scheduler, self.wd_scheduler = opt_constructor(self.scheduler,
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

    def vali(self, dataset_, epo, testing = False):
        self.model.eval()
        all_mse = []
        all_mae = []
        with torch.no_grad():
            for i, (x, y, _) in enumerate(dataset_):
                input = x.to(self.device)######################
                y = y.to(self.device)

                y_full, y_freq, y_horizon_pred = self.model(input)

                if i == 0:
                    all_pred = y_horizon_pred
                    all_y = y
                else:
                    all_pred = torch.cat((all_pred, y_horizon_pred), dim = 0)
                    all_y = torch.cat((all_y, y), dim = 0)

        self.logger.info(f"size: {x.shape}") # To check if downsampled input is correctly made (if applied as in exp section 4.2)

        if (epo % 10) == 0 or testing:
            pass
            # self.log.log_forecasting_vis(y_horizon_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), name_ = f"{epo}" if not testing else "testing")
        if testing:
            pass
            # all_mse = ((all_pred - all_y)**2)[:,-1,:]
            # self.log.log_forecasting_error_vis(all_mse)
        MSE_ = F.mse_loss(all_pred,all_y).detach().cpu().item()
        MAE_ = F.l1_loss(all_pred,all_y).detach().cpu().item()

        return MSE_, MAE_

    def train(self):
        self.logger.info("======================TRAIN MODE======================")

        early_stopping = EarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset, logger=self.logger)
        train_steps = len(self.training_data)
        self.logger.info(f'train_steps: {train_steps}')
        for epoch in range(self.n_epochs):
            speed_t = []
            epoch_time = time.time()
            self.model.train()
            self.hyper_variables.training_set()
            for i, (x, y, f_fullspan) in enumerate(self.training_data):
                x = x.to(self.device)######################
                y = y.to(self.device)
                
                f_fullspan = f_fullspan.to(self.device)
                if self.lr_scheduler is not None:
                    _new_lr = self.lr_scheduler.step()
                if self.wd_scheduler is not None:
                    _new_wd = self.wd_scheduler.step()
                self.optimizer.zero_grad()
                
                per_itr_time = time.time()

                y_pred, y_freq, y_horizon_pred = self.model(x)
                TD_loss, FD_loss = self.model.criterion(pred_TD = y_horizon_pred, target_TD = y.detach(), 
                                                    
                                                    pred_FD = y_freq, target_FD = f_fullspan.detach(),

                                                    pred_TD_full = y_pred, target_TD_full = torch.cat((x, y), dim = 1).detach(),

                                                    loss_mode = self.hyper_variables.loss_type)

                loss = (TD_loss* self.lamda) + (FD_loss * (1.- self.lamda))

                loss.backward()
                self.optimizer.step()
                speed_t.append(time.time() - per_itr_time)
                
                self.loss_TD.update(TD_loss.item())
                self.loss_FD.update(FD_loss.item())
                self.loss_total.update(loss.item())
                
                # grad_stats_AC = grad_logger_spec(self.model.named_parameters(), prob_ = "forecaster_", off= False) 
                # grad_stats_conv_IMAG = grad_logger_spec(self.model.named_parameters(), prob_ = "ll_NFF", off= False) 
                # grad_stats_exp_real = grad_logger_spec(self.model.named_parameters(), prob_ = "projection_in", off= False)
                # grad_stats_1 = grad_logger_spec(self.model.named_parameters(), prob_ = "phi_INFF", off= False) 
                # grad_stats_2 = grad_logger_spec(self.model.named_parameters(), prob_ = "cv_mlp", off= False) 
                # grad_stats_3 = grad_logger_spec(self.model.named_parameters(), prob_ = "channel_mixer", off= False)
                # grad_stats_4 = grad_logger_spec(self.model.named_parameters(), prob_ = "FT_generator", off= False)

                # self.log.log_into_csv_(epoch+1,
                #                             i,
                #                             self.loss_TD.avg,
                #                             self.loss_FD.avg,
                #                             _new_lr if self.lr_scheduler is not None else 0.,
                #                             _new_wd if self.wd_scheduler is not None else 0.,
                #                             grad_stats_AC.avg,
                #                             grad_stats_conv_IMAG.avg,
                #                             grad_stats_exp_real.avg,
                                            
                #                             grad_stats_1.avg,
                #                             grad_stats_2.avg,
                #                             grad_stats_3.avg,
                #                             grad_stats_4.avg)
                if (i + 1) % 100 == 0:
                    self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}] & s/iter:{np.mean(speed_t): .5f}, left time: {np.mean(speed_t) * (train_steps - i): .5f}, Loss_TD:{self.loss_TD.avg: .4f} , Loss_FD:{self.loss_FD.avg: .4f} , Loss_total: {self.loss_total.avg: .4f}")
                

            self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}] & speed per epoch: {(time.time() - epoch_time): .5f}")
            
            MSE_, MAE_ = self.vali(self.val_data, epoch+1)
            self.hyper_variables.testing_set() # This alters m_t or m_f in model for testing (if applied)
            MSE_t, MAE_t = self.vali(self.testing_data, 1)
            self.logger.info(f"Epoch[{epoch+1}],  MSE: {MSE_: .5f} & MAE: {MAE_: .5f}")
            self.logger.info(f"Epoch[{epoch+1}], TESTING --  MSE: {MSE_t: .5f} & MAE: {MAE_t: .5f}")
            early_stopping(MSE_, MAE_,  self.model, self.model_save_path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
            self.logger.info("")

    def test(self):
        if os.path.exists(os.path.join(str(self.model_save_path), '_checkpoint.pth')):
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), '_checkpoint.pth')))
            self.logger.info("Best trained Model called:")
        else: 
            raise ImportError(self.logger.info("Loading checkpoint model failed"))
        self.model.eval()
        self.logger.info("======================TEST MODE======================")
        self.hyper_variables.testing_set() # This alters m_t or m_f in model for testing (if applied)
        MSE_, MAE_ = self.vali(self.testing_data, 0, testing= True)

        self.logger.info(f"Forecasting result - MSE: {MSE_} & MAE: {MAE_}")
    

    def complexity_test(self):
        self.model.eval()
        self.logger.info("======================Complexity======================")
        self.hyper_variables.testing_set() # This alters m_t or m_f in model for testing (if applied)
        self.model.eval()

        with torch.no_grad():
            for i, (x, y, _) in enumerate(self.testing_data):
                input = x.to(self.device)######################
                y = y.to(self.device)
                flops = FlopCountAnalysis(self.model, input)
                break;
        macs_g = flops.total() / (10 ** 9)
        # Print the results
        self.logger.info(f"Number of FLOPs (GFLOPs): {macs_g}G")
        self.logger.info(flops.by_module_and_operator())
        with torch.no_grad():
            for i, (x, y, _) in enumerate(self.testing_data):
                input = x.to(self.device)######################
                y = y.to(self.device)
                start_time = time.time()
                y_full, _, _, _ , _, _, _ = self.model(input)
                end_time = time.time()
                break;
        peak_memory_after = torch.cuda.max_memory_allocated(self.device)
        inference_time = end_time - start_time
        self.logger.info(f"inference time: {inference_time} seconds")
        peak_memory_usage = peak_memory_after - self.peak_memory_init
        self.logger.info(f"Peak memory usage during inference: {peak_memory_usage / (1024 * 1024)} MB")
    