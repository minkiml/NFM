import torch
import torch.nn as nn
import numpy as np
import logging
import os
import time

from AnomalyD.NFM_AD import model_constructor
from AnomalyD.data_factory.data_loader import get_loader_segment
from utils.vars_ import HyperVariables
from utils.logger_ import Value_averager

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
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset=dataset_name
        self.logger = logger
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, '_checkpoint.pth'))
        self.val_loss_min = val_loss


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

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset,
                                              logger=self.logger)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset,
                                              logger=self.logger)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset,
                                              logger=self.logger)
        
        self.loss_TD = Value_averager()
        self.loss_FD = Value_averager()
        self.loss_total = Value_averager()

        self.device = torch.device(f'cuda:{self.gpu_dev}' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"GPU (device: {self.device}) used" if torch.cuda.is_available() else 'cpu used')
        self.build_model()
        self.criterion = nn.MSELoss()

        for i, (input_data, labels) in enumerate(self.train_loader):
            self.logger.info(f"shape : {input_data.shape},     labels: {labels.shape}")
            break;
        if self.DSR == 1:
            self.logger.info("Conventional reconstruction")
        else:
            self.logger.info("Upsampled reconstruction")
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
                                            lft = bool(self.lft),
                                            
                                            lft_siren_dim_in = self.siren_in_dim,
                                            lft_siren_hidden = self.siren_hidden,
                                            lft_siren_omega = self.siren_omega,
                                            lft_norm= bool(self.lft_norm),
                                            tau= self.tau,
                                            
                                            loss_type= self.loss_type)
        self.model = model_constructor(self.hyper_variables,
                                       self.DSR,
                                       bool(self.masking))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) # TODO 

        if torch.cuda.is_available():
            self.model.to(self.device)
    def adjust_learning_rate(self, optimizer, epoch, lr_):
        lr_adjust = {epoch: lr_ * (0.95 ** ((epoch - 1) // 1))}
        # lr_adjust = {epoch: lr_ * (0.8 ** ((epoch - 1) // 1))} # for WADI only !!!!!!!!!!!!
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            self.logger.info(f'Updating learning rate to {lr}')

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                # input = input_data.float().to(self.device)
                # output, series, prior, _ = self.model(input)
                
                input = input_data.float().to(self.device)[:,0::self.DSR,:]
                # print(input.shape, self.win_size//4)

                y, y_freq = self.model(input)

                ###########################
                TD_loss, FD_loss = self.model.criterion(y, input_data.float().to(self.device).detach(),
                                        y_freq, 
                                        loss_mode= self.hyper_variables.loss_type,
                                        mode = "testing")
                rec_loss = TD_loss + FD_loss

                loss_1.append((rec_loss).item())
            
        return np.average(loss_1)

    def train(self):
        self.logger.info("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset, logger=self.logger)
        train_steps = len(self.train_loader)
        self.logger.info(f'train_steps: {train_steps}')
        # self._get_profile(self.model)
        for epoch in range(self.num_epochs):
            iter_count = 0
            speed_t = []
            epoch_time = time.time()
            self.model.train()
            self.hyper_variables.training_set()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)[:,0::self.DSR,:]######################

                y, y_freq = self.model(input)
                ###########################
                per_itr_time = time.time()

                TD_loss, FD_loss = self.model.criterion(y, input_data.float().to(self.device).detach(),
                                     y_freq, 
                                     loss_mode= self.hyper_variables.loss_type,
                                     mode = "training")

                loss = TD_loss + FD_loss
                # loss1.backward(retain_graph=True)
                loss.backward()
                self.optimizer.step()
                speed_t.append(time.time() - per_itr_time)

                self.loss_TD.update(TD_loss.item())
                self.loss_FD.update(FD_loss.item())
                self.loss_total.update(loss.item())
                if (i + 1) % 100 == 0:
                    self.logger.info(f"epoch[{epoch+1}/{self.num_epochs}] & s/iter:{np.mean(speed_t): .5f}, left time: {np.mean(speed_t) * (train_steps - i): .5f}, Loss_TD:{self.loss_TD.avg: .4f} , Loss_FD:{self.loss_FD.avg: .4f}")
                    iter_count = 0

            self.logger.info(f"epoch[{epoch+1}/{self.num_epochs}] & speed per epoch: {(time.time() - epoch_time): .5f}")
            
            # vali_loss1 = self.vali(self.test_loader)
            self.hyper_variables.testing_set()
            vali_loss1 = self.vali(self.vali_loader)
            self.logger.info(f"Epoch[{epoch+1}] & Steps: {train_steps} & Train Loss: {self.loss_total.avg} & Vali Loss: {vali_loss1}")

            early_stopping(vali_loss1,  self.model, path)
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
        temperature = 50
        self.hyper_variables.testing_set()
        self.logger.info("======================TEST MODE======================")
        mode_ = "pointwise"#"pointwise"
        # (1) stastic on the train set
        attens_energy = []
        val_labels = []
        attens_energy2 = []
        for i, (input_data, labels) in enumerate(self.train_loader):
  
            input = input_data.float().to(self.device)[:,0::self.DSR,:]###################

            y, y_freq = self.model(input)
            TD, FD = self.model.criterion(y, input_data.float().to(self.device).detach(),
                                 y_freq,
                                 self.hyper_variables.loss_type,
                                 reduction = 'none',
                                 mode = "testing")
            # Point-wise
            if mode_ == "pointwise":
                TD_loss = TD # (B, L)
                FD_loss = 0.
        
            rec_loss = TD_loss + FD_loss
            cri = rec_loss.mean(-1)
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            val_labels.append(labels)
        # print(attens_energy)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold 
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.vali_loader): # thre_loader
            input = input_data.float().to(self.device)[:,0::self.DSR,:]###################
            y, y_freq = self.model(input)

            ###########################

            TD, FD = self.model.criterion(y, input_data.float().to(self.device).detach(),
                                 y_freq,
                                 self.hyper_variables.loss_type,
                                 reduction = 'none',
                                 mode = "testing")
            # Point-wise
            if mode_ == "pointwise":
                TD_loss = TD.mean(-1) # (B, L)
                FD_loss = 0.
            rec_loss = TD_loss + FD_loss
            cri = rec_loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            val_labels.append(labels)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        # thresh = test_energy.std(-1) * 2
        self.logger.info(f"train_energy: {train_energy.mean()}, test_energy{test_energy.mean()}")
        self.logger.info(f"Threshold : {thresh}")

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        attens_energy2 = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)[:,0::self.DSR,:]###################
            y, y_freq = self.model(input)

            TD, FD = self.model.criterion(y, input_data.float().to(self.device).detach(),
                                 y_freq,
                                 self.hyper_variables.loss_type,
                                 reduction = 'none',
                                 mode = "testing")
            # Point-wise(ADFORMER)
            if mode_ == "pointwise":
                TD_loss = TD # (B, L)
                FD_loss = 0.
            rec_loss = TD_loss + FD_loss

            cri = rec_loss.mean(-1)
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

            if mode_ == "segwise":
                if labels.dim() == 3:
                    labels = labels.squeeze(-1)
                test_labels.append((torch.sum(labels, dim = 1)>0)) # (B)
            else: test_labels.append(labels) # (B, L)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        self.logger.info(f"test_energy{test_energy.mean()}")
        self.logger.info(f"test_energy: {test_energy.shape}, test_labels{test_labels.shape}")
        
        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        self.logger.info(f"pred: {pred.shape}")
        self.logger.info(f"gt: {gt.shape}")

        # detection adjustment
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        self.logger.info(f"pred: {pred.shape}")
        self.logger.info(f"gt: {gt.shape}")

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        self.logger.info( f"Accuracy : {accuracy: .4f}, Precision : {precision: .4f}, Recall : {recall: .4f}, F-score : {f_score: .4f} ")
        return accuracy, precision, recall, f_score
