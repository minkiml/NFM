import os
import argparse
import logging
import numpy as np
import torch
from torch.backends import cudnn

from AnomalyD.AD_solver import Solver

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
    
    if config.mode == 'train':
        solver.train()
        pass
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100) 
    parser.add_argument('--input_c', type=int, default=38) 
    parser.add_argument('--output_c', type=int, default=38) 
    parser.add_argument('--batch_size', type=int, default=1024) 
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--DSR', type=int, default=2, help = "m_f value. If set to 1, it is full reconstruction") 
    parser.add_argument("--channel_dependence", type=int, default=0, help = "channel_dependence -- 1: True, 0: False (channel independent)")
    parser.add_argument('--masking', type=int, default=0)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument("--gpu_dev", type=str, default="6")
    parser.add_argument("--description", type=str, default='', help="optional")
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--log_path",
                        type=str,
                        default='./AnomalyD/Logs_/logs_',
                        help="path to save all texts")
    parser.add_argument("--run_id",
                        type=int,
                        default=0)
    # NFM
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--loss_type", type=str, default="TD", help= "TD: time domain loss, FD: Frequency domain loss, TFD: Mixture")
    # NFM params
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--inff_siren_hidden", type=int, default=64)
    parser.add_argument("--inff_siren_omega", type=int, default=30)
    parser.add_argument("--hidden_factor", type=int, default=3)
    parser.add_argument("--freq_span", type=int, default=-1)
    parser.add_argument("--layer_num", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--filter_type", type=str, default="INFF", choices=["INFF", "FNO", "AFNO", "GFN", "AFF"])
    parser.add_argument("--lft_norm", type=int, default=0, help = "Whether to apply normalization to input spectrum in LFT")
    parser.add_argument("--tau", type=str, default="independent", choices= ["independent", "shared"])

    # LFT (based on siren) params
    parser.add_argument("--lft", type=int, default=1)
    parser.add_argument("--siren_hidden", type=int, default=48)
    parser.add_argument("--siren_in_dim", type=int, default=4)
    parser.add_argument("--siren_omega", type=float, default=30.)

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
            for formatting testing data. If same as 'vars_in_train' then, conventional scenario")

    config = parser.parse_args()
    ##########################################################
    ##########################################################
    # global logger
    log_path = config.log_path + config.dataset + "_" + f"{config.run_id}" + f"_{config.description}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    os.environ["log_loc"] = f"{log_path}"
    root_dir = os.getcwd() 
    logging.basicConfig(filename=os.path.join(root_dir, f'{log_path}/log_all.txt'), level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('In main')
    logger.info(f"Experiment: Anomaly detection")
    
    config.model_save_path = os.path.join(log_path,"checkpoints") 

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        logger.info('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
