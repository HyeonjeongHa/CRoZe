import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from collections import defaultdict
import torch
import pandas as pd
import scipy.stats as stats
import pickle
import numpy as np

from nasbench_space.models import *
from nasbench_space.robust_nasbench201 import RobustnessDataset
from zero_cost_methods.pruners import *
from zero_cost_methods.dataset import *
from zero_cost_methods.corrupt_dataset import *


def check_nan(data):
    return torch.sum(torch.isnan(torch.tensor(data)))

    
def get_hrs(clean_acc, robust_acc):
    '''
    HRS = 2CR / (C+R)
    '''
    hrs = (2 * clean_acc * robust_acc) / (clean_acc + robust_acc)
    return hrs


def get_acc_proxy(args, proxy, proxies, robust_data, results):
    test_accs, robust_fgsm_4, robust_fgsm_8, hrs_fgsm_4, hrs_fgsm_8 = [], [], [], [], []
    robust_cc_weather = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    robust_cc_noise = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    robust_cc_blur = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    robust_cc_digital = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]

    _proxies = defaultdict(list)
    for i in range(args.end):

        # NASWOT contains NaN values
        is_nans = check_nan(proxies[proxy][i])
        if is_nans: continue
        _proxies[proxy].append(proxies[proxy][i])
        clean_acc = results[args.dataset]['clean']['accuracy'][robust_data.get_uid(i)]
        robust_fgsm_8_ = results[args.dataset]['fgsm@Linf']['accuracy'][robust_data.get_uid(i)][robust_data.meta["epsilons"]["fgsm@Linf"].index(8.0)]
        robust_fgsm_4_ = results[args.dataset]['fgsm@Linf']['accuracy'][robust_data.get_uid(i)][robust_data.meta["epsilons"]["fgsm@Linf"].index(4.0)]
        
        test_accs.append(clean_acc)
        robust_fgsm_8.append(robust_fgsm_8_)
        robust_fgsm_4.append(robust_fgsm_4_)
        hrs_fgsm_8.append(get_hrs(clean_acc, robust_fgsm_8_))
        hrs_fgsm_4.append(get_hrs(clean_acc, robust_fgsm_4_))
        
        if args.dataset == 'ImageNet16-120': continue
        for cc_type in RobustnessDataset.keys_cc:
            cc_accs = results[args.dataset][cc_type]['accuracy'][robust_data.get_uid(i)]
            if cc_type in WEATHER:
                _robust_cc = robust_cc_weather
            elif cc_type in NOISE:
                _robust_cc = robust_cc_noise
            elif cc_type in BLUR:
                _robust_cc = robust_cc_blur
            elif cc_type in DIGITAL:
                _robust_cc = robust_cc_digital
            else:
                _robust_cc = None

            for level, acc in enumerate(cc_accs):
                if _robust_cc is not None:
                    _robust_cc[level][cc_type].append(acc)

    return test_accs, robust_fgsm_4, robust_fgsm_8, hrs_fgsm_4, hrs_fgsm_8, robust_cc_weather, robust_cc_noise, robust_cc_blur, robust_cc_digital, _proxies


def get_corr_cc_single(robust_cc, proxy, proxies):
    corrs = []
    for cc_type in robust_cc[0].keys():
    
        corr_total = 0
        for level in range(5):
            corr = round(stats.spearmanr(robust_cc[level][cc_type], proxies[proxy])[0], 3)
            corr_total += corr
        
        corrs.append(corr_total/5)
    return round(sum(corrs)/len(corrs), 3)


def get_corr_nasbench201(args, proxies):

    robust_data = RobustnessDataset(path=args.rob_api_loc)
    results = robust_data.query(
        data = [args.dataset],
        measure = ['accuracy'],
        key = RobustnessDataset.keys_clean + RobustnessDataset.keys_adv + RobustnessDataset.keys_cc
    )
    
    corr_dict = defaultdict(list)
    save_dir = f'./{args.save_dir}/corr/{args.end}'
    os.makedirs(save_dir, exist_ok=True)

    f = open(os.path.join(save_dir, 'corr.txt'), 'w')

    corr_dict['name'].append('test_acc')
    corr_dict['name'].append('fgsm8')
    corr_dict['name'].append('fgsm4')
    corr_dict['name'].append('hrs_fgsm8')
    corr_dict['name'].append('hrs_fgsm4')
    if args.dataset != 'ImageNet16-120':
        corr_dict['name'].append('cc_weather')
        corr_dict['name'].append('cc_noise')
        corr_dict['name'].append('cc_blur')
        corr_dict['name'].append('cc_digital')
    
    for zero_cost_proxy in args.ZERO_COST_PROXY_LIST:
        test_accs, robust_fgsm_4, robust_fgsm_8, hrs_fgsm_4, hrs_fgsm_8, robust_cc_weather, robust_cc_noise, robust_cc_blur, robust_cc_digital, _proxies = get_acc_proxy(args, zero_cost_proxy, proxies, robust_data, results)
        corr_dict[zero_cost_proxy].append(round(stats.spearmanr(test_accs, _proxies[zero_cost_proxy])[0], 3))
        corr_dict[zero_cost_proxy].append(round(stats.spearmanr(robust_fgsm_8, _proxies[zero_cost_proxy])[0], 3))
        corr_dict[zero_cost_proxy].append(round(stats.spearmanr(robust_fgsm_4, _proxies[zero_cost_proxy])[0], 3))
        corr_dict[zero_cost_proxy].append(round(stats.spearmanr(hrs_fgsm_8, _proxies[zero_cost_proxy])[0], 3))
        corr_dict[zero_cost_proxy].append(round(stats.spearmanr(hrs_fgsm_4, _proxies[zero_cost_proxy])[0], 3))

        if args.dataset != 'ImageNet16-120':
            for _robust_cc in [robust_cc_weather, robust_cc_noise, robust_cc_blur, robust_cc_digital]:
                corr_dict[zero_cost_proxy].append(get_corr_cc_single(_robust_cc, zero_cost_proxy, _proxies))

    corr_df = pd.DataFrame(corr_dict, columns=list(corr_dict.keys()))
    file_name = '_'.join(args.ZERO_COST_PROXY_LIST)+'_results.csv'
    corr_df.to_csv(os.path.join(save_dir, file_name))

    return corr_df

   
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("corr")
    parser.add_argument('--search_space', type=str, default='nasbench201')
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
    parser.add_argument('--save_dir', type=str, default='./exp/', help='output directory')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--proxy_types', type=str, default='grad_norm')
    parser.add_argument('--rob_api_loc', default='', type=str, help='path to robust API for NASBench201')
    parser.add_argument('--proxy_path', type=str, default=None)
    args = parser.parse_args()

    if args.proxy_types == 'baselines':
        args.ZERO_COST_PROXY_LIST = ['grad_norm', 'snip', 'grasp', 'fisher', 'synflow']
    else:
        args.ZERO_COST_PROXY_LIST = args.proxy_types.split(',')
    print(args.ZERO_COST_PROXY_LIST)

    if args.end == 0: args.end = 15625     
    proxies = defaultdict(list)
    with open(args.proxy_path, "rb") as f:
        for i in range(args.start, args.end):
            data = pickle.load(f)
            for proxy in args.ZERO_COST_PROXY_LIST:
                proxies[proxy].append(data['logmeasures'][proxy])

    get_corr_nasbench201(args, proxies)
    