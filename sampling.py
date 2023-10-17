import os
import torch
import random
import time
import pickle
import utils
import numpy as np
import argparse
from nas_201_api import NASBench201API as API

from nasbench_space.models.nasbench2 import get_model_from_arch_str
from darts_space.genotypes import Genotype
from darts_space.model import NetworkCIFAR as Network
from zero_cost_methods.dataset import get_num_classes
from proxy import zero_cost_proxy


DARTS_PRIMITIVES = [
    'none',
    'avg_pool_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_7x1_1x7'
]

def mutate_architecture(steps, primitives, prev):

    select_step = torch.randint(1, 4, (1,)).item()
    def _mutate_cell(prev_gene):
        gene = prev_gene
        op_len = len(prev_gene)
        mutate_op = torch.randint(0, op_len, (1,)).item()
        op = primitives[torch.randint(len(primitives), (1,)).item()]
        while op == 'none' or (op == prev_gene[mutate_op][0]):
            op = primitives[torch.randint(len(primitives), (1,)).item()]
        
        gene[mutate_op] = (op, gene[mutate_op][1])
        return gene

    normal_cell = _mutate_cell(prev['normal'])
    reduction_cell = _mutate_cell(prev['reduction'])
    num_concat = torch.randint(1, steps+1, (1,)).item()  # Generate random num_concat

    if select_step == 1:
        normal_concat = torch.randint(2, 2 + steps, (num_concat,)).tolist()
    else:
        normal_concat = prev['normal_concat']
    
    if select_step == 2:
        reduction_concat = torch.randint(2, 2 + steps, (num_concat,)).tolist()
    else:
        reduction_concat = prev['reduction_concat']
    
    return {
        'normal': normal_cell,
        'reduction': reduction_cell,
        'normal_concat': normal_concat,  # Updated line
        'reduction_concat': reduction_concat,  # Updated line
    }


def random_architecture(steps, primitives):

    num_concat = torch.randint(1, steps+1, (1,)).item()  # Generate random num_concat
    def _random_cell():
        gene = []
        n_nodes = steps
        for i in range(n_nodes):
            for j in range(2+i):
                op = primitives[torch.randint(len(primitives), (1,)).item()]
                while op == 'none':
                    op = primitives[torch.randint(len(primitives), (1,)).item()]
                gene.append((op, j))
        return gene

    return {
        'normal': _random_cell(),
        'reduction': _random_cell(),
        'normal_concat': torch.randint(2, 2 + steps, (num_concat,)).tolist(),  
        'reduction_concat': torch.randint(2, 2 + steps, (num_concat,)).tolist(),  
    }


def check_duplicate(archs, arch):

    for archs_dup in archs:
        if archs_dup["normal"] == arch["normal"]:
            if archs_dup["reduction"] == arch["reduction"]:
                if archs_dup["normal_concat"] == arch["normal_concat"]:
                    if archs_dup["reduction_concat"] == arch["reduction_concat"]:
                        return False
                    else:
                        continue
                else:
                    continue
            else:
                continue
        else:
            continue
    return True


def mutate_sample_arch(args, prev_arch=None):

    num_classes = get_num_classes(args)
    if args.search_space == 'darts':
        steps = 4
        sampled_arch_list = []
        if prev_arch is None:
            for i in range(args.sample_pool):
                while True:
                    arch = random_architecture(steps, DARTS_PRIMITIVES)
                    genotype = Genotype(
                                normal=arch["normal"], normal_concat=arch["normal_concat"],
                                reduce=arch["reduction"], reduce_concat=arch["reduction_concat"],
                            )
                    model = Network(args.init_channels, num_classes, args.layers, args.auxiliary, args.drop_path_prob, genotype)
                    model_size = utils.count_parameters_in_MB(model)
                    if model_size > 6:
                        continue
                    else:
                        break
                sampled_arch_list.append(arch)
        else:
            for i in range(args.sample_pool):
                while True:
                    arch = mutate_architecture(steps, DARTS_PRIMITIVES, prev_arch)
                    genotype = Genotype(
                                normal=arch["normal"], normal_concat=arch["normal_concat"],
                                reduce=arch["reduction"], reduce_concat=arch["reduction_concat"],
                            )
                    model = Network(args.init_channels, num_classes, args.layers, args.auxiliary, args.drop_path_prob, genotype)
                    model_size = utils.count_parameters_in_MB(model)
                    if model_size > 6:
                        continue
                    else:
                        break
                sampled_arch_list.append(arch)
        return sampled_arch_list


def random_sample_arch(args, archs_dup = None):

    num_classes = get_num_classes(args)
    if args.search_space == 'nasbench201':
        archs = API(args.api_loc)
        total_indices = len(archs)
        real_indices = []
        if 'v1_0' in args.api_loc and args.dataset == 'cifar10':
            args.dataset = 'cifar10-valid'

        if not archs_dup is None:
            while True:
                sampled_indices = set(np.random.choice(total_indices, 1, replace=False))    
                while (sampled_indices in archs_dup):
                    sampled_indices = set(np.random.choice(total_indices, 1, replace=False))
                arch_str = archs[sampled_indices[0]]
                model = get_model_from_arch_str(arch_str, num_classes)
                # Model & Optimizer setting
                model = model.cuda()
                info = archs.query_meta_info_by_index(idx)
                model_size = info.get_compute_costs(args.dataset)['params']
                if model_size > 6:
                    continue
                else:
                    real_indices.append(idx)
                    break
        else:
            sampled_indices = set(np.random.choice(total_indices, int(args.sample_num*5), replace=False))
            cnt = 0
            
            for idx in sampled_indices:
        
                arch_str = archs[idx]
                model = get_model_from_arch_str(arch_str, num_classes)
                model = model.cuda()
                info = archs.query_meta_info_by_index(idx)
                model_size = info.get_compute_costs(args.dataset)['params']
                if model_size > 6:
                    continue
                real_indices.append(idx)
                cnt += 1
                if cnt>=args.sample_num:
                    break
                
        sampled_indices = np.array(real_indices) 
        with open(os.path.join(args.save_dir, f'random_architectures_{args.sample_num}_nasbench.pkl'), "wb") as f:
            pickle.dump(sampled_indices, f)
        return sampled_indices

    elif args.search_space == 'darts':
        steps = 4
        s = time.time()
        random_architecture(steps, DARTS_PRIMITIVES)
        t = time.time()
        
        cnt = 0 
        random_archs = []
        while True:
            arch = random_architecture(steps, DARTS_PRIMITIVES)
            genotype = Genotype(
                        normal=arch["normal"], normal_concat=arch["normal_concat"],
                        reduce=arch["reduction"], reduce_concat=arch["reduction_concat"],
                    )
            model = Network(args.init_channels, num_classes, args.layers, args.auxiliary, args.drop_path_prob, genotype)
            model_size = utils.count_parameters_in_MB(model)
            if model_size > 6:
                continue
            if not archs_dup is None:
                print('not none')
                verif = check_duplicate(archs_dup, arch)
                if verif:
                    random_archs.append(arch)
                    break
            else:
                verif = check_duplicate(random_archs, arch)
                if verif:
                    cnt += 1
                    random_archs.append(arch)
                if cnt >= args.sample_num:
                    break
                if cnt % 100 == 0:
                    print(cnt)

        if archs_dup is None:
            with open(os.path.join(args.save_dir, f'random_architectures_{args.sample_num}_darts.pkl'), "wb") as f:
                pickle.dump(random_archs, f)
        return random_archs
    
    else:
        NotImplementedError()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("random sampling")
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--search_space', type=str, default='darts', help='search space (darts/nasbench201)')
    parser.add_argument('--sample_num', type=int, default=300, help='number of sampled architecture in search space')
    parser.add_argument('--api_loc', default='', type=str, help='path to API')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
    parser.add_argument('--save_dir', default='./sample_arch', type=str)
    args = parser.parse_args()
    
    os. makedirs(args.save_dir, exist_ok=True)
    random_sample_arch(args)
