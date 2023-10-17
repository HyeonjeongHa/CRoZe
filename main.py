import torch
import argparse
import pickle
import os
import numpy as np
import time

from sampling import random_sample_arch, mutate_sample_arch
from proxy import zero_cost_proxy
from corr import get_corr_nasbench201
from train import train


def sample_arch(args):
    
    archs_dup, test = [], []
    best_arch, best_arch_val, prev_arch = None, 0, None

    os.makedirs(args.save_dir, exist_ok=True)
    sample_pool = args.sample_pool
    start_time = time.time()
    cnt = 0
    while cnt <= args.sample_num:
    
        if args.sampling_type == 'random':
            arch_list = random_sample_arch(args, archs_dup)
        elif args.sampling_type == 'mutate':
            if cnt == 0: args.sample_pool = args.init_pool
            else: args.sample_pool = sample_pool
            arch_list = mutate_sample_arch(args, prev_arch)
        elif args.sampling_type == 'warmup':
            if cnt == 0:
                args.sample_pool = args.init_pool
                arch_list = mutate_sample_arch(args, prev_arch)
            else:
                args.sample_pool = sample_pool
                arch_list = random_sample_arch(args, archs_dup)

        test = arch_list
        print(arch_list)
        if not prev_arch is None and (args.sampling_type != 'mutate'):
            if (args.sampling_type == 'warmup' and cnt == 0):
                print('pass')
            else:
                test.append(prev_arch)
        measures_save = zero_cost_proxy(args, test)
        
        value = measures_save[args.proxy_types]
        archs_dup = archs_dup + arch_list
        max_idx = int(np.argmax(value))
        max_arch_val = value[max_idx]
        prev_arch = test[max_idx]

        if best_arch_val < max_arch_val:
            best_arch = prev_arch
            best_arch_val = max_arch_val

        cnt += len(arch_list)
        if cnt % 50 == 0:
            print(f'cnt: {cnt}, arch: {prev_arch}')
            with open(os.path.join(args.save_dir, 'sample_history.txt'), 'a+') as f:
                f.write(f'cnt: {cnt}, arch: {prev_arch}, value: {max_arch_val}\n')
                f.close() 
                
    end_time = time.time()
    print(f'proxy:{args.proxy_types}\ncnt: {cnt}, prev_arch: {best_arch}')
    with open(os.path.join(args.save_dir, 'sample_history.txt'), 'a+') as f:
        f.write(f'total time: {end_time-start_time}\n')
        f.write(f'best arch: {best_arch}, proxy: {best_arch_val}')
        f.close()

    return best_arch


def main(args):

    if args.proxy_types == 'baselines':
        args.ZERO_COST_PROXY_LIST = ['grad_norm', 'snip', 'grasp', 'fisher', 'synflow']
    else:
        args.ZERO_COST_PROXY_LIST = args.proxy_types.split(',')
    print(args.ZERO_COST_PROXY_LIST)
    
    if args.w_sampling:
        assert len(args.ZERO_COST_PROXY_LIST) == 1
        # Search the best arch by sampling under guidance of proxy
        best_arch = sample_arch(args)

        # Best arch e2e
        if args.e2e:
            args.batch_size = 64
            args.weight_decay = 1e-4
            args.arch_type = 'darts_sampled'
            train(args, best_arch, proxy_name=args.proxy_types)
    else:
        assert args.search_space == 'nasbench201'
        # Calculate the proxy
        measures_save = zero_cost_proxy(args)
            
        # Calculate the correlation
        corr_df = get_corr_nasbench201(args, measures_save)
        
        # Best arch e2e
        if args.e2e:
            best_archs = dict()
            for proxy, value in measures_save.items():
                best_archs[proxy] = int(np.argmax(value))
            
            for proxy, best_arch in best_archs.items():
                args.batch_size = 64
                args.weight_decay = 1e-4
                args.arch_type = str(best_arch)
                train(args, best_arch, proxy_name=proxy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--save_dir', type=str, default='./exp/', help='output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for calculating proxies')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')

    # data load
    parser.add_argument('--data', type=str, default='', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')

    # sampling
    parser.add_argument('--search_space', type=str, default='darts', help='search space (darts/nasbench201)')
    parser.add_argument('--sample_num', type=int, default=100000, help='number of sampled architecture in search space')
    parser.add_argument('--w_sampling', action='store_true', default=False, help='use random sampling')
    parser.add_argument('--sampling_type', type=str, default='random', help='sampled pool')
    parser.add_argument('--sample_pool', type=int, default=5, help='sampled pool')
    parser.add_argument('--init_pool', type=int, default=50, help='sampled pool')
    parser.add_argument('--randomly_sampled_arch_path', type=str, default=None)

    # zero cost proxy
    parser.add_argument('--api_loc', default='', type=str, help='path to API for NASBench201')
    parser.add_argument('--rob_api_loc', default='', type=str, help='path to robust API for NASBench201')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--noacc', default=False, action='store_true', help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    parser.add_argument('--proxy_types', type=str, default='grad_norm')

    # e2e
    parser.add_argument('--e2e', action='store_true')
    parser.add_argument('--arch_type', type=str, default=None)
    parser.add_argument('--lr_scheduler', type=str, default='linear', help=['linear', 'cosine'])
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    parser.add_argument('--epsilon', type=float, default=8./255., help='perturbation')
    parser.add_argument('--num_steps', type=int, default=7, help='perturb number of steps')
    parser.add_argument('--step_size', type=float, default=2./255., help='perturb step size')
    parser.add_argument('--adv_loss', type=str, default='', help=' / pgd / fgsm')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--wandb_project_name', type=str)
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_attack_type', type=str, default='pgd')
    parser.add_argument('--eval_cc', action='store_true')
    parser.add_argument('--severity', type=int, default=None)
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    main(args) 
