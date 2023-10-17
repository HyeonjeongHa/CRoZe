import pdb
import pickle
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
import time
import json

from zero_cost_methods.pruners import *
from zero_cost_methods.dataset import *
from zero_cost_methods.weight_initializers import init_net
from nasbench_space.models import *
from darts_space.genotypes import Genotype
from darts_space.model import NetworkCIFAR as Network


def zero_cost_proxy(args, sampled_arch = None):
    
    if (sampled_arch is None):
        if args.search_space == 'darts':
            assert args.randomly_sampled_arch_path is not None
            # Do random sampling in sampling.py first
            # ex) "./sample_arch/random_architectures_"+str(args.sample_num)+"_darts.pkl"
            with open(args.randomly_sampled_arch_path, "rb") as f:
                random_archs = pickle.load(f)
            args.end = len(random_archs) if args.end == 0 else args.end

        elif args.search_space =='nasbench201':
            if args.noacc:
                random_archs = pickle.load(open(args.api_loc,'rb'))
            else:
                from nas_201_api import NASBench201API as API
                random_archs = API(args.api_loc)
            args.end = len(random_archs) if args.end == 0 else args.end
        
    else:
        args.end = len(sampled_arch)
        if args.search_space == 'darts':
            random_archs = sampled_arch

        elif args.search_space == 'nasbench201':
            if args.noacc:
                random_arch_list = pickle.load(open(args.api_loc,'rb'))
            else:
                from nas_201_api import NASBench201API as API
                random_arch_list = API(args.api_loc)
            random_archs = [random_arch_list[sampled_arch]]
        
        else:
            raise NotImplementedError

    pbar = tqdm(range(args.end))
    
    torch.manual_seed(args.seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_loader, val_loader = get_dataloaders(
        args.batch_size, args.batch_size, args.dataset, args.num_data_workers, datadir=args.data
    )
    num_class = get_num_classes(args)

    cached_res = []
    pre = 'cf' if 'cifar' in args.dataset else 'im'
    proxy_lists = '_'.join(args.ZERO_COST_PROXY_LIST)
    pfn = f'{args.search_space}{args.end}_{proxy_lists}{pre}{num_class}_seed{args.seed}_dl{args.dataload}_dlinfo{args.dataload_info}_initw{args.init_w_type}_initb{args.init_b_type}.p'
    op = os.path.join(args.save_dir, pfn)
    os.makedirs(args.save_dir, exist_ok=True)

    measures_save = defaultdict(list)
    for i in pbar:
        if i < args.start:
            continue
        if i >= args.end:
            break 

        if args.search_space == 'darts':
            arch_str = random_archs[i]
            
            genotype = Genotype(
                normal=arch_str["normal"], normal_concat=arch_str["normal_concat"],
                reduce=arch_str["reduction"], reduce_concat=arch_str["reduction_concat"],
            )
            net = Network(args.init_channels, num_class, args.layers, args.auxiliary, args.drop_path_prob, genotype)
        
        elif args.search_space == 'nasbench201':
            arch_str = random_archs[i]
            net = nasbench2.get_model_from_arch_str(arch_str, num_class)

        res = {'i': i, 'arch': arch_str}
        net.to(args.device)
            
        measures = predictive.find_measures(net, 
                                            train_loader, 
                                            (args.dataload, args.dataload_info, num_class),
                                            args.device,
                                            measure_names=args.ZERO_COST_PROXY_LIST,
                                            search_space=args.search_space)

        if measures == None:
            res['logmeasures'] = None
        else:
            for proxy, value in measures.items():
                measures_save[proxy].append(value)
            res['logmeasures'] = measures

        if args.search_space == 'nasbench201':
            info = random_archs.get_more_info(i, 'cifar10-valid' if args.dataset=='cifar10' else args.dataset, iepoch=None, hp='200', is_random=False)

            trainacc = info['train-accuracy']
            valacc   = info['valid-accuracy']
            testacc  = info['test-accuracy']
        
            res['trainacc'] = trainacc
            res['valacc'] = valacc
            res['testacc'] = testacc
        
        cached_res.append(res)

        # write to file
        if i % args.write_freq == 0 or i == args.end-1:
            print(f'writing {len(cached_res)} results to {op}')
            pf = open(op, 'ab')
            for cr in cached_res:
                pickle.dump(cr, pf)
            pf.close()
            cached_res = []
            
    return measures_save #, op
    

