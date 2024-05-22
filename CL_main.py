import os
import argparse
import logging
import numpy as np
import torch
from torch.backends import cudnn
from Classification.CL_solver import Solver

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
    solver.train()
    solver.test()
    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./Classification/Logs_/logs_', help="path to save all the products from each trainging")
    parser.add_argument("--id_", type=int, default=0, help="Run id")
    parser.add_argument("--data_path", type=str, default='./data', help="path to grab data")
    parser.add_argument("--description", type=str, default='', help="optional")
    parser.add_argument("--dataset", type=str, default="SpeechCommands")
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
    parser.add_argument("--patience", type=int, default=30)

    # optimizer
    parser.add_argument("--lr_", type=float, default=5e-4, help= "learning rate")
    parser.add_argument("--loss_type", type=str, default="TFDR", help= "Not used")
    parser.add_argument("--scheduler", type=int, default=0)
    ### Scheduler params
    parser.add_argument("--warm_up", type=float, default=0.2, help="portion of warm up given number of epoches, e.g., 20 percent by defualt")
    parser.add_argument("--start_lr", type=float, default=1e-5, help="starting learning rate")
    parser.add_argument("--ref_lr", type=float, default=1.5e-4, help= "peak learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-5, help = "final learning rate")
    parser.add_argument("--start_wd", type=float, default=0., help = "starting weight decay. 0. means no decay")
    parser.add_argument("--final_wd", type=float, default=0., help = "fianl weight decay")

    # NFM params
    parser.add_argument('--input_c', type=int, default=7, help = "Input channel dim") 
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--inff_siren_hidden", type=int, default=32)
    parser.add_argument("--inff_siren_omega", type=float, default=30.)
    parser.add_argument("--hidden_factor", type=int, default=3)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--filter_type", type=str, default="INFF", choices=["INFF", "FNO", "AFNO", "GFN", "AFF"])
    parser.add_argument("--init_xaviar", type=int, default=1, help = "Initialization method for linear projection")
    
    # LFT params
    parser.add_argument("--lft", type=int, default=1, help = "whether to use LFT")
    parser.add_argument("--siren_hidden", type=int, default=32)
    parser.add_argument("--siren_in_dim", type=int, default=4)
    parser.add_argument("--siren_omega", type=float, default=30.)
    parser.add_argument("--ff_std", type=int, default=128)
    parser.add_argument("--lft_norm", type=int, default=1, help = "Whether to apply normalization to input spectrum in LFT")
    parser.add_argument("--tau", type=str, default="independent", choices= ["independent", "shared"])

    # Classification params
    parser.add_argument("--sr_train", type=int, default=1, help = "sampling rate at training time. e.g., 1, 2, 4")
    parser.add_argument("--sr_test", type=int, default=1, help = "sampling rate at testing time. e.g., 1, 2, 4")
    parser.add_argument("--dropped_rate", type=int, default=0, help = "iregular setting with dropping. e.g., 30, 50, 70") 
    parser.add_argument("--mfcc", type=int, default=1, help="processed (1:True) or raw (0:False)") 
    parser.add_argument("--num_class", type=int, default=10)
    parser.add_argument("--CE_smoothing_scheduler", type=int, default=0)
    parser.add_argument("--channel_dependence", type=int, default=1, help = "1: True, 0: False (channel independent)")
    parser.add_argument("--freq_span", type=int, default=-1, help = "-1 for operating on full frequency span")

    # IN-Out for training
    parser.add_argument(
        "--vars_in_train",
        nargs='+',
        type=int,
        default=[360, 360, 96, 96],
        help="A set of variables [Fs, F_in, L_out (horizon), L_in (lookback)] for formatting training data")
    # IN-Out for testing
    parser.add_argument(
        "--vars_in_test",
        nargs='+',
        type=int,
        default=[360, 360, 180, 360],
        help="A set of variables [Fs, F_in, L_out (horizon), L_in (lookback)] \
            for formatting testing data. If same as 'vars_in_train' then, the task is conventional scenario")

    config = parser.parse_args()
    ##########################################################
    ##########################################################
    # global logger
    la = "_raw" if not config.mfcc else "_mfcc"
    log_path = config.log_path + config.dataset + la + "_" + f"{config.id_}" + "_" + f"{config.description}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    os.environ["log_loc"] = f"{log_path}"
    root_dir = os.getcwd() 
    logging.basicConfig(filename=os.path.join(root_dir, f'{log_path}/log_all.txt'), level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('In main')
    logger.info(f"Experiment: Time-series classification")
    
    config.model_save_path = os.path.join(log_path,"checkpoints") 
    config.plots_save_path = os.path.join(log_path,"plots") 
    config.his_save_path = os.path.join(log_path,"hist") 
    args = vars(config)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        logger.info('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
