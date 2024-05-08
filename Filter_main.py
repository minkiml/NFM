import os
import argparse
import logging
import numpy as np
import torch
from torch.backends import cudnn

from Filtering.Filter_learner import Solver

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))    
    solver.fit()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./Filtering/Logs_/logs_', help="path to save all the products from each trainging")
    parser.add_argument("--id_", type=int, default=0, help="Run id")
    parser.add_argument("--data_path", type=str, default='./data', help="this doesn't matter here")
    parser.add_argument("--description", type=str, default='', help="optional")
    parser.add_argument("--dataset", type=str, default="sim-filter", choices=["sim-filter"])

    # Save path
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--plots_save_path', type=str, default='plots')
    parser.add_argument('--his_save_path', type=str, default='hist')
    # Training params
    parser.add_argument("--plotting", type=int, default=0, help = "0: False and 1: True")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--gpu_dev", type=str, default="6")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--batch_testing", type=int, default=32)

    # optimizer
    parser.add_argument("--lr_", type=float, default=5e-4, help= "Peak learning rate")
    parser.add_argument("--loss_type", type=str, default="TD", help= "TD: time domain loss, FD: Frequency domain loss, TFD: Mixture, TFDR")
    parser.add_argument("--scheduler", type=int, default=0, help= "Whether to use optimizer scheduler for lr and wd")
    ### Scheduler params
    parser.add_argument("--warm_up", type=float, default=0.2, help="portion of warm up given number of epoches, e.g., 20 percent by defualt")
    parser.add_argument("--start_lr", type=float, default=1e-5, help="starting learning rate")
    parser.add_argument("--ref_lr", type=float, default=1.5e-4, help= "Peak learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-5, help = "final learning rate")
    parser.add_argument("--start_wd", type=float, default=0.04, help = "starting weight decay")
    parser.add_argument("--final_wd", type=float, default=0.4, help = "fianl weight decay")

    # NFM params
    parser.add_argument('--input_c', type=int, default=7)  
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--inff_siren_ff_dim", type=int, default=256)
    parser.add_argument("--inff_siren_hidden", type=int, default=64)
    parser.add_argument("--inff_siren_omega", type=int, default=30)
    parser.add_argument("--hidden_factor", type=int, default=3)
    parser.add_argument("--layer_num", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--filter_type", type=str, default="INFF", choices=["INFF", "FNO", "AFNO", "GFN", "AFF"])
    # LFT (based on siren) params
    parser.add_argument("--siren_hidden", type=int, default=48)
    parser.add_argument("--siren_in_dim", type=int, default=4)
    parser.add_argument("--siren_omega", type=float, default=30.)

    # Filter
    parser.add_argument("--max_modes", type=int, default=400)
    parser.add_argument("--diversity", type=int, default=1)
    parser.add_argument("--num_modes", type=int, default=10)
    parser.add_argument("--filter_mode", type=str, default="Lowpass", choices=["Lowpass", "Bandpass", "Highpass"])
    parser.add_argument("--num_class", type=int, default=10)
    parser.add_argument("--channel_dependence", type=int, default=1, help = "1: True, 0: False (channel independent)")
    parser.add_argument("--freq_span", type=int, default=-1)

    # IN-Out for training
    parser.add_argument(
        "--vars_in_train",
        nargs='+',
        type=int,
        default=[360, 360, 96, 96],
        help="A set of variables [F_L, F_N, L (horizon), N (lookback)] for formatting training data")
    # IN-Out for testing
    parser.add_argument(
        "--vars_in_test",
        nargs='+',
        type=int,
        default=[360, 360, 180, 360],
        help="A set of variables [F_L, F_N, L (horizon), N (lookback)] \
            for formatting testing data. If same as 'vars_in_train' then, conventional scenario")

    config = parser.parse_args()
    ##########################################################
    ##########################################################

    # global logger
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    log_path = config.log_path + config.dataset + "_" + f"{config.id_}"+ "_" f"{config.description}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    os.environ["log_loc"] = f"{log_path}"
    root_dir = os.getcwd() 
    logging.basicConfig(filename=os.path.join(root_dir, f'{log_path}/log_all.txt'), level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('In main')
    logger.info(f"Experiment: INFF")
    
    config.model_save_path = os.path.join(log_path,"checkpoints") 
    config.plots_save_path = os.path.join(log_path,"plots") 
    config.his_save_path = os.path.join(log_path,"hist") 
    args = vars(config)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        logger.info('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
