import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# Set plt params
sns.set(style='ticks', font_scale=1.2)
plt.rcParams['figure.figsize'] = 12,8

class Logger(object):
    '''
    Handy logger object to log every training histories. 
    '''
    def __init__(self,
                 plot_path, # log path
                 hist_path,
                 *argw):
        self.log_plot_path = plot_path
        self.log_train_path = hist_path
        if argw is not None:
            # CSV
            self.types = []
            # -- print headers
            with open(os.path.join(self.log_train_path, "training_log.csv"), '+a') as f:
                for i, v in enumerate(argw, 1):
                    self.types.append(v[0])
                    if i < len(argw):
                        print(v[1], end=',', file=f)
                    else:
                        print(v[1], end='\n', file=f)

    def log_into_csv_(self, *argw):
        # logging losses, lr, etc, that are generated at every epoch or iteration into csv
        with open(os.path.join(self.log_train_path, "training_log.csv"), '+a') as f:
            for i, tv in enumerate(zip(self.types, argw), 1):
                end = ',' if i < len(argw) else '\n'
                print(tv[0] % tv[1], end=end, file=f)

    def log_stats_into_csv_(self, *argw):
        # logging losses, lr, etc, that are generated at every epoch or iteration into csv

        with open(os.path.join(self.log_train_path, "stats.csv"), '+a') as f:
            f.write(f"{argw[0].label}_mean,{argw[0].label}_std,{argw[0].label}_L2,{argw[1].label}_mean,{argw[1].label}_std,{argw[1].label}_L2,{argw[2].label}_mean,{argw[2].label}_std,{argw[2].label}_L2\n")
            for val1_mean, val1_std, val1_norm, val2_mean, val2_std, val2_norm, val3_mean, val3_std, val3_norm in zip(argw[0].mean_list, argw[0].std_list, argw[0].norm_list,
                                        argw[1].mean_list, argw[1].std_list, argw[1].norm_list,
                                        argw[2].mean_list, argw[2].std_list, argw[2].norm_list):
                f.write(f"{val1_mean.item()},{val1_std.item()},{val1_norm.item()},{val2_mean.item()},{val2_std.item()},{val2_norm.item()},{val3_mean.item()},{val3_std.item()},{val3_norm.item()}\n")
   
    # get more functions on demand
    def log_pics(self, x, y, name_ = "", epo_ = 0):
        # Save 2d scatters (rep)
        fig = plt.figure(figsize=(12, 8), 
          dpi = 600) 
        axes = fig.subplots()
        scatter = axes.scatter(x = x[:,0], y= x[:,1], c = y[:], 
                    s=15, cmap="Spectral", alpha = 0.8)# , edgecolors= "black" 

        # Get unique class labels
        unique_classes = np.unique(y[:, 0])

        # Iterate through each class and mark one point
        for ii, class_label in enumerate(unique_classes):
            class_indices = np.where(y[:, 0] == class_label)[0]
            sample_index = class_indices[10]  # Choose the first sample for marking
            axes.text(x[sample_index, 0], x[sample_index, 1], f'{ii}', color='black', fontsize=15, ha='center', va='center', fontweight='bold')

        cbar = plt.colorbar(scatter)
        cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=20)
        # color_bar = plt.colorbar()
        # color_bar.set_label('Domain Prediction', rotation=270, labelpad=20, fontsize=20)

        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        plt.savefig(os.path.join(self.log_plot_path,name_ + f"_{epo_}.png" )) 
        plt.clf()   
        plt.close(fig)

    # get more functions on demand
    def log_forecasting_vis(self, pred, ground_t, gt_ext = None, name_ = "", last_ = False):
        B, L, C = pred.shape
        c_ = 1 if last_ else C
        # assert pred.shape == ground_t.shape
        for i in range (c_ if c_ < 10 else 7):
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            axes.plot(pred[-1, :,i], color = "red",
                alpha = 0.8, label = 'NFM')
            axes.plot(ground_t[-1, :,i], color = "blue",
                alpha = 0.8, label = 'Ground truth')
            if gt_ext is not None:
                axes.plot(gt_ext[-1, :,i], color = "green",
                    alpha = 0.8, label = 'Ground truth full')
            plt.xticks(fontweight='bold', fontsize = 20)   
            plt.yticks(fontweight='bold', fontsize = 20)
            legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1))

            plt.savefig(os.path.join(self.log_plot_path, name_ + f"_feature {i}_" + ".png" ), bbox_inches='tight') 
            plt.clf()   
            plt.close(fig) 
    # get more functions on demand
    def log_forecasting_error_vis(self, errors):
        td_ = errors.T.detach().cpu().numpy()
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        im = axes.imshow(td_, cmap='viridis', aspect='auto')
        # Set axis labels and title
        plt.xlabel("seq")
        plt.ylabel("feature")
        # Add a colorbar using the ScalarMappable
        cbar = plt.colorbar(im)
        # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
        plt.savefig(os.path.join(self.log_plot_path, "Forecasting_Error.png")) 
        plt.clf()   
        plt.close(fig)
        
    def log_time_domain_vis(self, ground_t, name_ = ""):
        fig = plt.figure(figsize=(12, 8), 
        dpi = 600) 
        axes = fig.subplots()
        axes.plot(ground_t[:,-1], color = "blue",
            alpha = 0.8, label = 'Input X')
        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1))

        plt.savefig(os.path.join(self.log_plot_path, name_ + f"Input X_{name_}" + ".png" ), bbox_inches='tight') 
        plt.clf()   
        plt.close(fig) 
        # get more functions on demand
    def frequency_reponse(self, f = None, range = -1, name_ = ""):
        if f.dim() == 2:
            F, c = f.shape
            magnitude_response = 20 * np.log10(np.abs(f.cpu().detach().numpy()))
        else:
            B, F, c = f.shape
            magnitude_response = 20 * np.log10(np.abs(f.cpu().detach().numpy()) + 1e-4).mean(axis=0)
        magnitude_response[magnitude_response <= -3] = - 40
        fig = plt.figure(figsize=(12, 8), 
        dpi = 600) 
        axes = fig.subplots()
        # axes.plot(freq_bins, magnitude_response.mean(axis = 1) if c > 1 else magnitude_response[:,0], color='blue')
        im = axes.imshow(magnitude_response.T, aspect='auto', cmap='viridis')
        plt.title('Frequency response')
        plt.xlabel('Frequency Component (F)')
        plt.ylabel('Feature')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\mathbf{Gain (dB)}$', fontweight='bold', fontsize=25)
        # plt.xlim(0, freq_bins.shape[0] if range == -1 else range)
        # plt.grid()
        plt.savefig(os.path.join(self.log_plot_path,"Frequency_gain_" + f"{name_}" + ".png" )) 
        plt.clf()   
        plt.close(fig)
  
        fig = plt.figure(figsize=(12, 8), 
        dpi = 600) 
        axes = fig.subplots()
        # axes.plot(freq_bins, magnitude_response.mean(axis = 1) if c > 1 else magnitude_response[:,0], color='blue')
        axes.plot(magnitude_response.mean(1), color='blue')
        plt.title('Frequency response')
        plt.xlabel('Frequency Component (F)')
        plt.ylabel('Gain')
        plt.savefig(os.path.join(self.log_plot_path,"Frequency_gain1d_" + f"{name_}" + ".png" )) 
        plt.clf()   
        plt.close(fig)

    def log_confusion_matrix(self, cm, name_ = ""):
        fig = plt.figure(figsize=(12, 8), 
                dpi = 600) 
        axes = fig.subplots()
        # Display the attention map
        sns.heatmap(cm, cmap='viridis')
        # Set axis labels and title
        plt.xlabel("GT")
        plt.ylabel("Pred")
        plt.savefig(os.path.join(self.log_plot_path, "CM_" + f"{name_}" + ".png")) 
        plt.clf()   
        plt.close(fig)
    

    def log_2d_vis(self, x, f = None, name_ = "", time = False):
        if time == False:
            if x.dim() == 2:
                l, c = x.shape
                # time domain
                td_ = x.T.detach().cpu().numpy()
                # mag
                magnitude = torch.abs(f).cpu().detach().numpy()
                # phase
                phase = torch.angle(f).cpu().detach().numpy()
                # freq (= mag)
                imaginary =(f.imag).cpu().detach().numpy()
                real_ =(f.real).cpu().detach().numpy()
            else:
                b, l, c = x.shape
                td_ = x.mean(dim=0).T.detach().cpu().numpy()
                magnitude = torch.abs(f).mean(dim=0).cpu().detach().numpy()
                phase = torch.angle(f).mean(dim=0).cpu().detach().numpy()
                imaginary =(f.imag).mean(dim=0).cpu().detach().numpy()
                real_ =(f.real).mean(dim=0).cpu().detach().numpy()
            fig = plt.figure(figsize=(12, 8), 
                dpi = 600) 
            axes = fig.subplots()
            # Display the attention map
            im = axes.imshow(td_, cmap='viridis', aspect='auto')
            # Set axis labels and title
            plt.xlabel("t")
            plt.ylabel("feature")
            # Add a colorbar using the ScalarMappable
            cbar = plt.colorbar(im)
            # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
            plt.savefig(os.path.join(self.log_plot_path, "LFT_" + f"{name_}" + ".png")) 
            plt.clf()   
            plt.close(fig)

            # Magnitude
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.imshow(magnitude.T, aspect='auto', cmap='viridis')
            # plt.title('Magnitude Visualization')
            plt.xlabel('Frequency Component', fontweight='bold')
            plt.ylabel('Hidden Dimension', fontweight='bold')
            plt.xticks(fontweight='bold')   
            plt.yticks(fontweight='bold')
            cbar = plt.colorbar(im)
            cbar.set_label('Magnitude', fontweight='bold')
            # cbar.set_ticks(fontweight='bold')

            plt.savefig(os.path.join(self.log_plot_path,"magnitude_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)

            # Phase
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.imshow(phase.T, aspect='auto', cmap='twilight')
            plt.title('Phase Visualization')
            plt.xlabel('Frequency Component (F)')
            plt.ylabel('Feature Dimension (d)')
            cbar = plt.colorbar(im)
            plt.savefig(os.path.join(self.log_plot_path,"Phase_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)

            # Imaginary only
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.imshow(imaginary.T, aspect='auto', cmap='twilight')
            plt.title('Imaginary Values')
            plt.xlabel('Frequency Component (F)')
            plt.ylabel('Feature Dimension (d)')
            cbar = plt.colorbar(im)
            plt.savefig(os.path.join(self.log_plot_path,"Imaginary_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)

            # real only
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            im = axes.imshow(real_.T, aspect='auto', cmap='twilight')
            plt.title('Real Values')
            plt.xlabel('Frequency Component (F)')
            plt.ylabel('Feature Dimension (d)')
            cbar = plt.colorbar(im)
            plt.savefig(os.path.join(self.log_plot_path,"Real_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)

            # Frequency
            ff_ = magnitude.mean(axis = 1)
            ff_[0] *= 0.
            freq_bins = np.arange(magnitude.shape[0])
            fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
            axes = fig.subplots()
            axes.plot(freq_bins, ff_ if c > 1 else magnitude[:,0], color='blue')
            # plt.title('Frequency Domain Plot')
            plt.xlabel('Frequency Component', fontweight='bold')
            plt.ylabel('Magnitude', fontweight='bold')
            plt.xticks(fontweight='bold')   
            plt.yticks(fontweight='bold')
            plt.grid()
            plt.savefig(os.path.join(self.log_plot_path,"Mean_Frequency_" + f"{name_}" + ".png" )) 
            plt.clf()   
            plt.close(fig)
        else:
            td_ = x.mean(dim=0).T.detach().cpu().numpy()
            fig = plt.figure(figsize=(12, 8), 
                dpi = 600) 
            axes = fig.subplots()
            # Display the attention map
            im = axes.imshow(td_, cmap='viridis', aspect='auto')
            # Set axis labels and title
            plt.xlabel("t")
            plt.ylabel("feature")
            # Add a colorbar using the ScalarMappable
            cbar = plt.colorbar(im)
            # cbar.set_label(r'$\mathbf{z_c}$', fontweight='bold', fontsize=25)
            plt.savefig(os.path.join(self.log_plot_path, "LFT_" + f"{name_}" + ".png")) 
            plt.clf()   
            plt.close(fig)

class Value_averager(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count        
    @property
    def _get_avg(self):
        return self.avg
    
def grad_logger(named_params, prob_ = 'linears'):
    stats = Value_averager()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if prob_ in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats

def grad_logger_spec(named_params, prob_ = 'linears', off = False):
    stats = Value_averager()
    stats.first_layer = None
    stats.last_layer = None
    if not off:
        for n, p in named_params:
            if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
                pass
                if prob_ in n:
                    grad_norm = float(torch.norm(p.grad.data))
                    stats.update(grad_norm)
    return stats

class param_stats_tracker(object):
    def __init__(self, name_= ""):
        self.reset(name_)
    def reset(self, name_ ):
        self.norm_ = 0.
        self.mean = 0
        self.std= 0

        self.mean_history = []
        self.std_history = []
        self.norm_history = []
        self.label = name_
    def update(self, p_, n=1):
        try:
            self.mean = torch.mean(p_)
            self.std = torch.std(p_)
            self.norm_ = torch.norm(p_)

            self.mean_history.append(self.mean)
            self.std_history.append(self.std)
            self.norm_history.append(self.norm_)
        except Exception:
            pass 
    def logging_(self, named_params):
        prob_ = self.label
        for n, p in named_params:
            if not (n.endswith('.bias') or len(p.shape) == 1):
                if prob_ in n:
                    self.update(p.data)
    def out_logged(self):
        self.mean_list = torch.stack((self.mean_history), dim = 0)
        self.std_list = torch.stack((self.std_history), dim = 0)
        self.norm_list = torch.stack((self.norm_history), dim = 0)

