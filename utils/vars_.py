import os
import logging
import torch
import torch.fft as fft 
import numpy as np
class HyperVariables(object):
    '''
    This class is a handy class to control variables in one place
    '''
    def __init__(self,
                 sets_in_training,
                 sets_in_testing,
                 freq_span = -1,
                 C_ = 3,
                 multivariate = True,
                 revstat = False,
                 dropout = 0.1, 
                 droppath = 1, 
                 std = 0.06,
                 var_revin = True,

                 filter_type = "INFF",
                 hidden_dim = 64,
                 final_hidden_dim = 0,
                 hidden_factor = 3,
                 inff_siren_hidden = 64,
                 inff_siren_omega = 30,
                 layer_num = 1,
                 lft = True,
                 
                 lft_siren_dim_in = 16,
                 lft_siren_hidden = 48,
                 lft_siren_omega = 30,
                 loss_type = "TFD",
                 class_num = None):
        # Gloabl logger
        log_loc = os.environ.get("log_loc")
        root_dir = os.getcwd() 
        logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all'), level=logging.INFO,
                            format = '%(asctime)s - %(name)s - %(message)s')
        self.logger = logging.getLogger('From hyper_vars')

        self.loss_type = loss_type 
        self.class_num = class_num
        # Init params
        self.init_std = 0.06 
        self.init_mean = 0.
        
        # INR-based LFT params (if LFT_type = "naive" the below params are not used)
        self.LFT_siren_omega = lft_siren_omega
        self.LFT_siren_dim_in = lft_siren_dim_in
        self.LFT_siren_hidden = lft_siren_hidden 
        self.lft = lft
        # NFM and NFF Hidden layer params
        self.INFF_siren_hidden = inff_siren_hidden
        self.INFF_siren_omega = inff_siren_omega
        self.hidden_dim = hidden_dim
        self.final_hidden_dim = hidden_dim if final_hidden_dim == 0 else final_hidden_dim
        self.hidden_factor = hidden_factor
        self.filter_type = filter_type             
        self.layer_num = layer_num 

        self.dropout = dropout

        # Mode switching (training <--> testing) 
        if sets_in_training == sets_in_testing: 
            self.sets_in_training = sets_in_training
            self.sets_in_testing = sets_in_testing
        else:
            self.sets_in_training = sets_in_training
            self.sets_in_testing = sets_in_testing

        # channel indepdent or not 
        self.C_true = C_
        self.multivariate = multivariate
        self.C_ = self.C_true if self.multivariate else 1


        # Standard sampling frequency (output frequency)
        self.Fs = None
        
        # Variables for target (y)
        self.F_out = None
        self.L_out = None
        self.T_out = None

        # Variables for input (x)
        self.F_in = None
        self.L_in = None
        self.T_in = None # assume to be 1 (reference)
        self.f_in = None

        # Variables for output (model output {x & y})
        self.L_base = None# This is equivalent to the length of model output
        self.f_base = None
        self.m_factor = None
        self.freq_span = 0 
        self.scale_factor = 1.
        self.scale_factor_IN_to_OUT = 1.
        self.fspan = freq_span # one can set this other than -1 (default) to apply cut-off frequency (a heuristic low-pass filter)
        self.__set_vars__(Fs = self.sets_in_training[0], 
                          F_in = self.sets_in_training[1], 
                          L_out = self.sets_in_training[2], 
                          L_in = self.sets_in_training[3])

        # to print condtions and vars
        self.check_var_conditions()
        self.display_vars()

        # Hard-coded SR testing setting used in classification experiements (otherwise, not used)
        self.sets_in_testing2 = [self.Fs, int(self.F_in / 2), 0, int(self.F_in / 2)] # SR = 1/2
        self.sets_in_testing3 = [self.Fs, int(self.F_in / 4), 0, int(self.F_in / 4)] # SR = 1/4

    # Setters to default training and testing variables. 
    # For non-default setups, call __set_vars__ function with arguments.
    def training_set(self):
        self.__set_vars__(Fs = self.sets_in_training[0], 
                    F_in = self.sets_in_training[1], 
                    L_out = self.sets_in_training[2], 
                    L_in = self.sets_in_training[3])
    def testing_set(self):
        if self.sets_in_testing is not None:
            self.__set_vars__(Fs = self.sets_in_testing[0], 
                        F_in = self.sets_in_testing[1], 
                        L_out = self.sets_in_testing[2], 
                        L_in = self.sets_in_testing[3])
            # self.display_vars()
        else: pass

    # Hard-coded SR testing setting used in classification experiements (otherwise, not used)
    def testing_set2(self):
        if self.sets_in_testing2 is not None:
            self.__set_vars__(Fs = self.sets_in_testing2[0], 
                        F_in = self.sets_in_testing2[1], 
                        L_out = self.sets_in_testing2[2], 
                        L_in = self.sets_in_testing2[3])
    def testing_set3(self):
        if self.sets_in_testing3 is not None:
            self.__set_vars__(Fs = self.sets_in_testing3[0], 
                        F_in = self.sets_in_testing3[1], 
                        L_out = self.sets_in_testing3[2], 
                        L_in = self.sets_in_testing3[3])
            
    # Set hyper variables (can change anytime on-the-fly during training or testing time) 
    # and model immediately reacts correctly. 
    def __set_vars__(self, Fs, F_in, L_out, L_in):
        self.Fs = Fs
        
        self.F_out = Fs
        self.L_out = L_out
        self.T_out = self.__set_T__(self.L_out, self.F_out)

        self.F_in = F_in
        self.L_in = L_in
        self.T_in = self.__set_T__(self.L_in, self.F_in)  
        self.f_in = L_in // 2 + 1
        self.__set_fourier_base__()
        self.__set_freq_span__(self.fspan)
        self.__set_mapping__()
        # m_f (sqare root is due to the use of normalization in DFT)
        self.scale_factor =  np.sqrt(Fs / F_in)
        # m_t (sqare root is due to the use of normalization in DFT)
        self.scale_factor_IN_to_OUT = np.sqrt(self.m_factor) 
        # self.display_vars()
        self.check_var_conditions()

    def __set_T__(self, L, F):
        return L / F

    def __set_freq_span__(self, freq_span):
        if freq_span != -1:
            self.freq_span = freq_span
            self.L_span = (self.freq_span - 1) * 2
        else:
            self.freq_span = self.f_base
            self.L_span = self.L_base
    def __set_fourier_base__(self):
        self.L_base = int(np.ceil(self.Fs * (self.T_out + self.T_in)))
        self.f_base = int(self.L_base // 2 + 1)
        self.m_factor = (self.T_out / self.T_in) + 1.
    
    ##### Check valid conditions for processing  ####
    def check_var_conditions(self):
        assert self.T_out >= 0, "T_out must not be less than 0. If no predictive (future in time) \
                                output is required as in classification task for example, set T_out = 0"
    #### Mapping related functions ####
    def __set_mapping__(self):
        # the round operation could be any... floor, round, ceil
        self.m = torch.arange(0,self.freq_span, self.m_factor, requires_grad=False).floor().to(torch.int64) # (B, f_in*)
    def __get_mapping__(self, B, no_exp = False):
        # Get mapping from f_in to f_base
        if no_exp:
            return self.m
        else: return self.m.expand(B,-1) # (B, f_in*)
    
    def map_to_base(self, in_, lfb = None, freq = True):
        # This is slow implementation
        B, F, c = in_.shape # input space variable
        mapping = self.__get_mapping__(B).to(in_.device)
        if mapping.shape[-1] > F: # If downsampled case input
            mapping = mapping[:,:F]
        if freq:
            to_base = torch.complex(torch.zeros((B, self.f_base, c),requires_grad = True), 
                                torch.zeros((B, self.f_base, c),requires_grad = True)).to(in_.device)
        else:
            to_base = torch.zeros((B, self.f_base, c)).to(in_.device)
        base = to_base.scatter(1, mapping.unsqueeze(-1).expand(-1,-1,c), in_)
        if lfb is not None:
            base[:,:lfb.shape[1],:] += lfb
        return base
        
    def fourier_extension(self, in_, base): # change name to Fourier extension
        mapping = self.__get_mapping__(in_.shape[0], no_exp=True).to(in_.device)
        in_ = in_[:,:len(mapping),:]
        F = in_.shape[1] # input space variable
        if mapping.shape[-1] > F: # If downsampled case input
            mapping = mapping[:F]
        base[:,mapping,:] += in_
        return base

    def input_(self, x):
        # To handle channel indepence in minibatch
        B, L, C = x.shape
        assert L == self.L_in
        self.flag_out = False
        if self.multivariate == False and C != 1:
            self.flag_out = True
            x = x.permute(0,2,1).reshape(-1, self.C_, L).permute(0,2,1)
            B, L, C = x.shape
        return x

    def output_(self, y_freq):
        # To handle channel indepence in minibatch
        B, L_base, C = y_freq.shape
        if self.multivariate == False and self.flag_out:
            y_freq = y_freq.permute(0,2,1).reshape(-1, self.C_true, L_base).permute(0,2,1)
        y_freqf = self.DFT_(y_freq)
        return y_freq, y_freqf
    
    def DFT_(self, x, dim = 1):
        # FFT
        return fft.rfft(x, dim = dim, norm = "ortho") 
    
    def IDFT_(self, freq, L = None, dim = 1):
        return fft.irfft(freq, n = self.L_base if L is None else L, dim = dim, norm = "ortho") 
    
   
    def DFT_irre(self, x, time_loc, dim = 1):
        # DFT implementation to handle irregular TS (slow! and expensive!)
        # x (B, N, C) -> B, C, N
        # time_loc (B, N) --> assumed to be already sorted and it is in range (0 to 1)
        raise NotImplementedError("Not recommend to use it as computationally to expensive") 
        B, N, C = x.shape
        x = x.transpose(1,2).view(-1, N) # (B * C, N)
        timeInterval = self.T_in # assume it is in the interval [0, 1]
        N = x.shape[1]
        # frequency bin (pos conj - - DC - - neg conj)
        k = torch.arange(-np.floor(self.L_base / 2), np.floor((self.L_base - 1) / 2) + 1)
        
        # Compute DFT matrix for irregular time series
        D = (torch.exp(1j * 2 * torch.pi * (time_loc[:,:, None] / timeInterval) * k) / N).to(x.device) # B X N X K matrix 
        # Pseudo Inverse of D
        D = torch.linalg.pinv(D)[:,:self.f_base,:][:,::-1,:] # torch.linalg.pinv # B, K, N --> B, pos conj, N
        D = D.unsqueeze(1).expand(-1,C,-1,-1).view(B*C,-1,N).transpose(1, 2).detach() # B * C, N, pos conj
        # working out with only the pos conjugate
        freq = torch.bmm(x.unsqueeze(1), D) # (BC, 1, N) mm (BC, N, pos conj) --> (BC, 1, pos conj)
        return freq.view(B, C, -1).transpose(1,2) / torch.sqrt(N) # B, pos conj, C
    
    #### Displayer (manual check) ####
    def display_vars(self):
        self.logger.info("")
        self.logger.info("*******************************************")
        self.logger.info("Variable setting")
        self.logger.info(f"Input length L_in: {self.L_in}")
        self.logger.info(f"Desired output length L_out: {self.L_out}")
        self.logger.info("")
        self.logger.info(f"Standard sampling frequency Fs: {self.Fs}")
        self.logger.info("")
        self.logger.info(f"Input sampling frequency F_in: {self.F_in}")
        self.logger.info(f"Output sampling frequency F_out: {self.F_out}")
        self.logger.info("")
        self.logger.info(f"Input timespan (reference) T_in: {self.T_in}")
        self.logger.info(f"Output timespan T_out: {self.T_out}")
        self.logger.info("")
        self.logger.info(f"Given input frequency length f_in: {self.f_in}")
        self.logger.info(f"Required base length L_base: {self.L_base}")
        self.logger.info(f"Required base frequency spectrum length f_base: {self.f_base}")
        self.logger.info(f"Mapping factor f_in -> f_base : {self.m_factor}")
        self.logger.info("")
        self.logger.info(f"Scale factor scale(train->test) : {self.scale_factor}")
        self.logger.info("")
        self.logger.info(f"Frequency span (cutoff) : {self.freq_span}")
        self.logger.info(f"L span (ds) : {self.L_span}")
        self.logger.info("*******************************************")
        # TODO OTHER INFOS
        self.logger.info(f"Channel dim: {self.C_true}")
        self.logger.info(f"Processing channel dim: {self.C_}")
        self.logger.info("Multivariate" if self.multivariate else "Channel independent")
        self.logger.info(f"Revstat on" if self.revstat is not None else "Revstat off")

        self.logger.info(f"mapping length: {self.m.shape}")
        self.logger.info(f"mapping matrix: {self.m}")

        self.logger.info(f"Training sets: {self.sets_in_training}")
        self.logger.info(f"Testing sets: {self.sets_in_testing}")