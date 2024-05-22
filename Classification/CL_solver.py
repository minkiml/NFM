import torch
import numpy as np
import logging
import os
import time

from Classification.NFM_CL import model_constructor
from utils.optimizer import opt_constructor
from Classification.data_factory.TS_dataloader import Load_dataset
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
    def __init__(self, patience=7, verbose=False,dataset_name='', delta=0, logger = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = -np.Inf

        self.early_stop = False
        self.delta = delta
        self.dataset=dataset_name
        self.logger = logger

    def __call__(self, acc, model, path, label = "ACC"):
        if acc > self.best_acc:
            self.save_checkpoint(acc, model, path, label = label)
            self.best_acc = acc
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(f'No validation improvement & EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, acc, model, path,
                        label = "ACC"):
        if self.verbose:
            self.logger.info(f'Validation improved ({label}: {np.abs(self.best_acc):.6f} --> {np.abs(acc):.6f}).  Saving model ...')
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
        data = Load_dataset({"data_path": self.data_path,
                                    "sub_dataset": self.dataset,

                                    "sr_train": self.sr_train,
                                    "sr_test": self.sr_test,
                                    "mfcc": self.mfcc,
                                    "drop_rate": self.dropped_rate,

                                    "batch_training": self.batch,
                                    "batch_testing": self.batch_testing})
        
        self.training_data = data.__get_training_loader__()
        self.testing_data = data.__get_testing_loader__()
        self.val_data = data.__get_val_loader__() 
        if self.dataset == "SpeechCommands" and self.mfcc == 0:
            self.testing_data_sr2 = data.__get_testing2_loader__()
            self.testing_data_sr4 = data.__get_testing3_loader__()
        del data
        self.loss_CE = Value_averager()

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
                            ('%.5f', 'loss_CE'),
                            ('%.5f', 'lr'),
                            ('%.5f', 'wd'),
                            ('%.4e', 'INFF'), 
                            ('%.4e', 'cv_mlp'), 
                            ('%.4e', 'LTF'), 
                            ('%.4e', 'projection_in'),
                            ('%.4e', 'final_layer'),
                            ('%.4e', 'classifier'),
                            ('%.4e', 'channel mixing')
                            )
        self.count = 70 if self.mfcc == 1 else 50
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
                                            lft_norm= bool(self.lft_norm),
                                            tau= self.tau,

                                            lft_siren_dim_in = self.siren_in_dim,
                                            lft_siren_hidden = self.siren_hidden,
                                            lft_siren_omega = self.siren_omega,
                                            
                                            class_num = self.num_class,
                                            CE_smoothing_scheduler = bool(self.CE_smoothing_scheduler),
                                            ff_std= self.ff_std,
                                            init_xaviar= bool(self.init_xaviar))
        self.model = model_constructor(self.hyper_variables)

        ipe = len(self.training_data)
        self.optimizer, self.lr_scheduler, self.wd_scheduler = opt_constructor(self.scheduler,
                                                                            self.model,
                                                                            lr = self.lr_,

                                                                            warm_up = self.n_epochs* ipe * self.warm_up , #int(self.n_epochs* ipe * self.warm_up),
                                                                            fianl_step = self.n_epochs* ipe, #int(self.n_epochs* ipe),
                                                                            start_lr = self.start_lr,
                                                                            ref_lr = self.ref_lr,
                                                                            final_lr = self.final_lr,
                                                                            start_wd = self.start_wd,
                                                                            final_wd = self.final_wd)
        self.model.smoothing_rate_reset(warmup_steps = self.n_epochs* ipe * 0.1 , #int(self.n_epochs* ipe * self.warm_up),
                                        T_max = self.n_epochs* ipe, #int(self.n_epochs* ipe),
                                        start_rate = 0.4,
                                        ref_rate = 0.4,
                                        final_rate = 0.15)
        if torch.cuda.is_available():
            self.model.to(self.device)

    def vali(self, dataset_, epo, testing = False):
        with torch.no_grad():
            for i, (x, y) in enumerate(dataset_):
                input = x.to(self.device) 
                y = y.to(self.device)
                self.model.eval()
                y_pred  = self.model.inference(input)
                
                if i == 0:
                        all_pred = y_pred
                        all_y = y
                else:
                    all_pred = torch.cat((all_pred, y_pred), dim = 0)
                    all_y = torch.cat((all_y, y), dim = 0)
        all_pred = all_pred.detach().cpu().numpy()
        all_y = all_y.detach().cpu().numpy()
        num_ = all_pred.shape[0]

        ACC_ = (all_pred == all_y).sum() / num_ 
        false_indices = all_pred != all_y
        false_indices = torch.nonzero(torch.tensor(false_indices)).squeeze()
        return ACC_

    def train(self, finetuning = False):
        if finetuning: # TODO: del this as not supported in this code
            self.logger.info("======================FINETUNING MODE======================")
            if os.path.exists(os.path.join(str(self.model_save_path), '_checkpoint.pth')):
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(str(self.model_save_path), '_checkpoint.pth')))
                self.logger.info("Best trained Model called:")
                self.model.melt_layers(layers = None)
                self.model.freeze_layers(layers = ["reconstructor"])

            else:
                raise ImportError("Loading failed. No pretrained model exists")
        else:
            self.logger.info("======================TRAIN MODE======================")

        early_stopping = EarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset, logger=self.logger)
        train_steps = len(self.training_data)
        self.logger.info(f'train_steps: {train_steps}')
        # self._get_profile(self.model)

        for epoch in range(self.n_epochs):
            speed_t = []
            epoch_time = time.time()
            self.hyper_variables.training_set()
            for i, (x, y) in enumerate(self.training_data):
                input = x.to(self.device)######################
                y = y.to(self.device)
                self.model.train()
                if self.lr_scheduler is not None:
                    _new_lr = self.lr_scheduler.step()
                if self.wd_scheduler is not None:
                    _new_wd = self.wd_scheduler.step()
                self.optimizer.zero_grad()

                per_itr_time = time.time()
                y_pred = self.model(input)
                CE_loss = self.model.criterion_cl(logit = y_pred, target = y.detach())

                loss = CE_loss
                loss.backward()
                self.optimizer.step()
                speed_t.append(time.time() - per_itr_time)

                self.loss_CE.update(CE_loss.item())
                # grad_stats_AC = grad_logger_spec(self.model.named_parameters(), prob_ = "phi_INFF") 
                # grad_stats_conv_1 = grad_logger_spec(self.model.named_parameters(), prob_ = "cv_mlp") 
                # grad_stats_conv_2 = grad_logger_spec(self.model.named_parameters(), prob_ = "FT_generator") 
                # grad_stats_exp_real = grad_logger_spec(self.model.named_parameters(), prob_ = "projection_in")
                # grad_stats_3 = grad_logger_spec(self.model.named_parameters(), prob_ = "phi_INFF")
                # grad_stats_5 = grad_logger_spec(self.model.named_parameters(), prob_ = "classifier")
                # grad_stats_6 = grad_logger_spec(self.model.named_parameters(), prob_ = "channel_mixer")
                # self.log.log_into_csv_(epoch+1,
                #                             i,
                #                             self.loss_CE.avg,
                #                             _new_lr if self.lr_scheduler is not None else 0.,
                #                             _new_wd if self.wd_scheduler is not None else 0.,
                #                             grad_stats_AC.avg,
                #                             grad_stats_conv_1.avg,
                #                             grad_stats_conv_2.avg,
                #                             grad_stats_exp_real.avg,
                #                             grad_stats_3.avg,
                #                             grad_stats_5.avg,
                #                             grad_stats_6.avg)
                if (i + 1) % 100 == 0:
                    self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}] & s/iter:{np.mean(speed_t): .5f}, left time: {np.mean(speed_t) * (train_steps - i): .5f}, Loss_CE:{self.loss_CE.avg: .4f}")

            self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}] & speed per epoch: {(time.time() - epoch_time): .5f}")
            self.hyper_variables.testing_set()
            ACC_ = self.vali(self.val_data, epoch+1)
            self.logger.info(f"Epoch[{epoch+1}],  ACC: {ACC_: .5f}")
            self.logger.info("")
            early_stopping(ACC_, self.model, self.model_save_path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

    def test(self):
        if os.path.exists(os.path.join(str(self.model_save_path), '_checkpoint.pth')):
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), '_checkpoint.pth')))
            self.logger.info("Best trained Model called:")
        self.model.eval()
        self.logger.info("======================TEST MODE======================")
        self.hyper_variables.testing_set()
        ACC = self.vali(self.testing_data, 0, testing= True)
        self.logger.info(f"Classification result SR = 1 -> ACC: {ACC} ")
        if self.dataset == "SpeechCommands" and self.mfcc == 0:
            print(self.mfcc)
            self.hyper_variables.testing_set2()
            ACC2 = self.vali(self.testing_data_sr2, 0, testing= False)
            self.hyper_variables.testing_set3()
            ACC3 = self.vali(self.testing_data_sr4, 0, testing= False)
            self.logger.info(f"")
            self.logger.info(f"Classification result SR = 2 -> ACC: {ACC2} ")
            self.logger.info(f"")
            self.logger.info(f"Classification result SR = 4 -> ACC: {ACC3} ")

    def complexity_test(self):
        self.model.eval()
        self.logger.info("======================Complexity Test======================")
        self.hyper_variables.testing_set() 
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.testing_data):
                input = x.to(self.device)######################
                y = y.to(self.device)
                flops = FlopCountAnalysis(self.model, input)
                break;
        macs_g = flops.total() / (10 ** 9)
        # Print the results
        self.logger.info(f"Number of FLOPs (GFLOPs): {macs_g}G")
        self.logger.info(flops.by_module_and_operator())
        with torch.no_grad():
            for i, (x, y) in enumerate(self.testing_data):
                input = x.to(self.device)######################
                y = y.to(self.device)
                start_time = time.time()
                y_pred = self.model(input)
                end_time = time.time()
                break;
        peak_memory_after = torch.cuda.max_memory_allocated(self.device)
        inference_time = end_time - start_time
        self.logger.info(f"inference time: {inference_time} seconds")

        peak_memory_usage = peak_memory_after - self.peak_memory_init
        self.logger.info(f"Peak memory usage during inference: {peak_memory_usage / (1024 * 1024)} MB")