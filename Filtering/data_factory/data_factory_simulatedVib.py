import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging

class filter_fitting_dataset(Dataset):
    def __init__(self, 
            set_ = [],
            data_name = "",
            type_ = "training"):
        super(filter_fitting_dataset, self).__init__()

        # Basic info
        self.name = data_name
        self.type_ = type_

        self.train_x = None
        self.train_y = None
        self.train_freq_target = None
        self.set_dataset(set_[0], set_[1], set_[2], set_[3], set_[4])
    def set_dataset(self, x, y, ff, noise_, clean_wave):
        self.train_x = x
        self.train_y = y
        self.train_ff = ff
        self.train_noise_ = noise_
        self.train_clean_wave = clean_wave
    def __len__(self):
        return self.train_x.shape[0] 
    def __getitem__(self, index):            
        return self.train_x[index], self.train_y[index], \
            self.train_ff[index], self.train_noise_[index], self.train_clean_wave[index]

class Synthetic_data(object):
    def __init__(
                # Loaded dataset
                self, 
                path = "",
                sub_dataset = "sim-vib",
                varset_train = [1000, 1000, 0, 1000], # [Fs, F_in, L_out (horizon), L_in (lookback)]
                max_modes = 200,
                diversity = 1,
                num_modes = 5,
                
                type_ = "Lowpass",
                num_class = 10,
                batch_training = 100):
        # Global logger
        log_loc = os.environ.get("log_loc")
        root_dir = os.path.abspath(os.path.join(os.getcwd(), '..')) # go to one root above
        logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all'), level=logging.INFO,
                            format = '%(asctime)s - %(name)s - %(message)s')
        self.logger = logging.getLogger('From Synthetic functions')

        self.batch_ = batch_training
        self.path = path
        self.sub_dataset = sub_dataset
        
        # some generation parameters 
        self.Fs = varset_train[0]
        self.L_in = varset_train[3]
        self.max_fs_features = max_modes if max_modes <= int(self.Fs / 2) else int(self.Fs / 2) # following Nyquist frequency fs/2
        self.duration = diversity
        self.number_of_modes = num_modes
        
        self.diversity = 1 # number of different sets of modes (compositions)
        self.num_class = num_class
        self.type_ = type_
        self.varset_train = varset_train
        self.__read_and_construct__()

    def __read_and_construct__(self):
        # Check if the dataset already exists in the root directory

        self.L_ = self.L_in

        trim_rate = 20 # To simply mitigate Gibbs phenomenon
        signals = []
        modes_ = []
        noises = []
        class_labels = []
        clearn_wave = []
        if self.sub_dataset == "sim-vib":
            for _ in range(self.diversity):
                signal, mode = get_sines(T = self.duration,
                                        trim_rate = trim_rate,
                                        L_in = self.L_in ,
                                        max_freqmode = self.max_fs_features,
                                        num_modes = self.number_of_modes,
                                        manual_modes = None,
                                        phase_= False,
                                        noise= True,
                                        c_ = self.C_)
                signals.append(signal)
                modes_.append(mode)
            signals = np.concatenate((signals), axis = 0)
            modes_ = np.concatenate((modes_), axis = 0)
            x = (
            torch.from_numpy(np.linspace(0, self.duration, self.L_in, endpoint=False))
            .type(torch.FloatTensor)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(self.diversity,1,1)
            ) # (div, length, 1) -> time index

            f = (
                torch.from_numpy(signals)
                .type(torch.FloatTensor)
            ) # (div, length, 1)

            ff = torch.fft.rfft(f, dim = 1, norm = "ortho")
            self.data_to_fit = (x, f, ff)
            self.logger.info(f"{self.diversity} simulated sine wave(s) (~vibration) was generated to fit.")

        elif self.sub_dataset == "sim-filter":
            self.C_ = 1
            for i in range(self.num_class):
                signal, mode, noise, wave = get_sines_with_classes(
                                        task_=self.type_,
                                        n=100,
                                        T=1,
                                        L_in=self.L_in,
                                        trim_rate=trim_rate,
                                        max_freqmode=self.max_fs_features,
                                        num_modes_N=self.number_of_modes,
                                        num_modes_K=self.number_of_modes * 4,
                                        phase_=False,
                                        noise=True,
                                        c_=1)
                class_label = np.ones((100, ), dtype =int) * i
                class_labels.append(class_label)
                clearn_wave.append(wave)
                signals.append(signal)
                modes_.append(mode)
                noises.append(noise)
            signals = np.concatenate((signals), axis = 0)
            modes_ = np.concatenate((modes_), axis = 0)
            noises = np.concatenate((noises), axis = 0)
            clearn_wave = np.concatenate((clearn_wave), axis = 0)
            class_labels = np.concatenate((class_labels), axis = 0)

            x = (
                torch.from_numpy(signals)
                .type(torch.FloatTensor)
            )
            clearn_wave = (
                torch.from_numpy(clearn_wave)
                .type(torch.FloatTensor)
            )
            noises = (
                torch.from_numpy(noises)
                .type(torch.FloatTensor)
            )

            y = (
                torch.from_numpy(class_labels)
                .type(torch.IntTensor)
            ) # (div, length, 1)
            ff = torch.fft.rfft(x, dim = 1, norm = "ortho")
            
            self.data_to_fit = DataLoader(filter_fitting_dataset(set_=[x.to(torch.float32), y.to(torch.int64), ff, noises, clearn_wave.to(torch.float32)],
                                            data_name = self.sub_dataset,
                                            type_ = "training"),
                                        batch_size = self.batch_,
                                        shuffle = True,
                                        num_workers= 10,
                                        drop_last=True)
            self.logger.info(f"class number: {self.num_class} data for learning filter.")


        self.testing_data = None
        self.val_loader = None


    def __get_training_loader__(self):
        return self.data_to_fit
    def __get_testing_loader__(self):
        return self.testing_data
    def __get_val_loader__(self):
        return self.val_loader
    
    def __get_info__(self): # TODO
        return (self.L_, self.C_)
    

def get_sines(T = 1,
              L_in = 1000,
              trim_rate = 20,
              max_freqmode = 500,
              num_modes = 5,
              manual_modes = None,
              phase_ = False,
              noise = False,
              c_ = 1):
    L_in = L_in + (trim_rate * 2)
    t = np.linspace(0, T, L_in, endpoint=False)

    if manual_modes is not None:
        modes = manual_modes
    else:
        modes = np.random.choice(max_freqmode, num_modes * c_, replace= False).reshape(1,-1, c_) #(num modes,)

    amp_ = np.random.uniform(0.3, 0.8, size = num_modes * c_)

    waves = (amp_.reshape(1,-1, c_) * np.sin(2 * np.pi * modes * t.reshape(-1,1,1) + \
                                         ((np.random.uniform(-1, 1, size = num_modes* c_) * np.pi).reshape(1,-1,c_) if phase_ else 0.) )).sum(1).reshape(1, -1, c_) \
            + (np.random.normal(0, 0.2, size = (1,L_in,1)) if noise else 0.)
    
    waves = (waves - waves.mean(1, keepdims = True)) / waves.std(1, keepdims = True) * 0.4# standardize 
    return waves[:,trim_rate:-trim_rate,:], modes

def get_sines_with_classes(
        task_='Lowpass',
        n=100,
        T=1,
        L_in=1000,
        trim_rate=20,
        max_freqmode=500,
        num_modes_N=5,
        num_modes_K=20,
        noise=False,
        c_=1,
        noise_var = 0.1):
    # n samples of each class of whole classes
    # Defined setup
    setups = {
        "Bandpass": [int(max_freqmode*0.4), int(max_freqmode*0.6)],
        "Lowpass": [3, int(max_freqmode * 0.2)], 
        "Highpass": [int(max_freqmode* 0.8), max_freqmode]
    }[task_]

    L_in = L_in + (trim_rate * 2) # To remove gibbs phenomenon
    t = np.linspace(0, T, L_in, endpoint=False)

    modes_N = np.repeat(np.random.choice(np.arange(setups[0], setups[1]), num_modes_N * c_, replace=False).reshape(1, -1, c_), n , axis = 0)  # (1, num modes_N, c)
    print(setups)
    if task_ == 'Lowpass':
        modes_K = [np.random.choice(np.arange(setups[1]+5, max_freqmode), num_modes_K * c_, replace=False).reshape(1, -1, c_) for _ in range(n)] # (1, num modes_N, c)
        modes_K = np.concatenate((modes_K), axis = 0)
    elif task_ == 'Highpass':
        modes_K = [np.random.choice(np.arange(5, setups[0]-5), num_modes_K * c_, replace=False).reshape(1, -1, c_) for _ in range(n)] # (1, num modes_N, c)
        modes_K = np.concatenate((modes_K), axis = 0)
    elif task_ == 'Bandpass':
        modes_K_1 = np.concatenate(([np.random.choice(np.arange(3, setups[0]-5), int(num_modes_K / 2), replace=False).reshape(1, -1, c_) for _ in range(n)]), axis = 0) # (1, num modes_N, c)
        modes_K_2 = np.concatenate(([np.random.choice(np.arange(setups[1]+5, max_freqmode-3), int(num_modes_K / 2) , replace=False).reshape(1, -1, c_) for _ in range(n)]), axis = 0) # (1, num modes_N, c)
        modes_K = np.concatenate((modes_K_1, modes_K_2), axis = 1)
    
    
    all_modes = np.concatenate((modes_N, modes_K), axis = 1)

    amp_N = np.repeat(np.random.uniform(0.3, 0.8, size=(1,num_modes_N, c_)), n, axis = 0)
    amp_K = np.random.uniform(0.3, 0.6, size=modes_K.shape )
    all_amp = np.concatenate((amp_N, amp_K), axis = 1)

    waves_ori = (all_amp * np.sin(2 * np.pi * all_modes * t.reshape(1, 1, -1))).sum(1).reshape(n, -1, 1) 
    # mean_ = waves_ori.mean(1, keepdims=True)
    # var_ = waves_ori.std(1, keepdims=True)
    # waves_ori = (waves_ori - mean_) / var_  # standardize

    if noise:
        noise = (np.random.normal(0, noise_var, size=waves_ori.shape) if noise else 0.)
        waves = waves_ori + noise
        noise = noise[:, trim_rate:-trim_rate, :]
    else: noise = None
    mean_ = waves.mean(1, keepdims=True)
    var_ = waves.std(1, keepdims=True)
    waves = (waves - mean_) / var_  # standardize
    return waves[:, trim_rate:-trim_rate, :], all_modes, noise,  waves_ori