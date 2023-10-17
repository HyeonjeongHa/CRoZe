import os
import numpy as np
import torch
import torchattacks
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.autograd import Variable

import utils
from train_utils import *
from zero_cost_methods.pruners import *
from zero_cost_methods.dataset import *
from zero_cost_methods.corrupt_dataset import *
from nasbench_space.models import *
from darts_space.genotypes import Genotype
from darts_space.model import NetworkCIFAR as Network
from attack.attack_lib import madry_loss, pgd_attack


ADV_RUSH = {'normal': [('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], 'normal_concat': range(2, 6), 'reduction': [('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], 'reduction_concat':range(2, 6)}
DRNAS_CIFAR = {'normal': [('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3)], 'normal_concat': range(2, 6), 'reduction': [('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('skip_connect', 4), ('sep_conv_5x5', 1)], 'reduction_concat': range(2, 6)}
DRNAS_IMAGENET = {'normal': [('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_3x3', 1)], 'normal_concat': range(2, 6), 'reduction': [('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)], 'reduction_concat': range(2, 6)}
PCDARTS_CIFAR = {'normal': [('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], 'normal_concat': range(2, 6), 'reduction': [('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], 'reduction_concat': range(2, 6)}
PCDARTS_IMAGENET = {'normal': [('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], 'normal_concat': range(2, 6), 'reduction': [('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], 'reduction_concat': range(2, 6)}
CROZE_CIFAR10 = {'normal': [('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_7x7', 4)], 'reduction': [('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('conv_7x1_1x7', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('conv_7x1_1x7', 0), ('sep_conv_7x7', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 0), ('sep_conv_7x7', 1), ('conv_7x1_1x7', 2), ('dil_conv_3x3', 3), ('sep_conv_3x3', 4)], 'normal_concat': [4, 5, 3, 4], 'reduction_concat': [3, 2, 5, 5]}
CROZE_CIFAR100 =  {'normal': [('max_pool_3x3', 0), ('sep_conv_7x7', 1), ('avg_pool_3x3', 0), ('sep_conv_7x7', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_7x7', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 3), ('skip_connect', 4)], 'reduction': [('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_7x7', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 3), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3), ('avg_pool_3x3', 4)], 'normal_concat': [2, 2, 4, 5], 'reduction_concat': [3, 4, 5, 5]}
CROZE_IMAGENET = {'normal': [('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_7x7', 4)], 'reduction': [('sep_conv_7x7', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_7x7', 3), ('skip_connect', 0), ('conv_7x1_1x7', 1), ('max_pool_3x3', 2), ('dil_conv_3x3', 3), ('avg_pool_3x3', 4)], 'normal_concat': [2, 5, 3, 4], 'reduction_concat': [3, 3, 4, 5]}


def test(args, best_net):
    
    num_class = get_num_classes(args)
    _, valid_queue = get_dataloaders(
        args.batch_size, args.batch_size, args.dataset, args.num_data_workers, normalize=False, datadir=args.data
    )
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Find the architecture in the sampled lists
    if args.search_space == 'darts':

        genotype = Genotype(
            normal=best_net["normal"], normal_concat=best_net["normal_concat"],
            reduce=best_net["reduction"], reduce_concat=best_net["reduction_concat"],
        )
        model = Network(args.init_channels, num_class, args.layers, args.auxiliary, args.drop_path_prob, genotype)
        
    elif args.search_space == 'nasbench201':
        
        from nas_201_api import NASBench201API as API
        api = API(args.api_loc)
        
        arch_str = api[best_net]
        model = nasbench2.get_model_from_arch_str(arch_str, num_class)
    
    else:
        NotImplementedError()

    args.main_path = f'{args.ckpt_dir}/_eval-{args.eval_attack_type}'
    os.makedirs(args.main_path, exist_ok=True)

    ckpt_path = os.path.join(f'{args.ckpt_dir}', 'checkpoint')
    ckpt_path = os.path.join(ckpt_path, 'best.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=False)

    model = model.cuda()
    model.eval()
    
    logger = utils.Logger(
        log_dir=args.main_path,
        exp_name=args.main_path.replace('/', '_'),
        exp_suffix='',
        write_textfile=True,
        use_wandb=False,
        wandb_project_name=args.wandb_project_name,
        entity=args.wandb_entity
    )
    logger.update_config(args, is_args=True)

    print(f'Infer with attack: {args.eval_attack_type}\nstep size:{args.step_size}\nepsilon: {args.epsilon}\nnum_steps: {args.num_steps}')
    top1, top5, objs, adv_top1, adv_top5, adv_objs = infer(args, 0, valid_queue, model, criterion)
    logger.write_log_nohead({
        'epoch': 0,
        'valid/adv/loss': adv_objs,
        'valid/adv/top1': adv_top1,
        'valid/adv/top5': adv_top5,
        'valid/clean/loss': objs,
        'valid/clean/top1': top1,
        'valid/clean/top5': top5,
    }, step=0)
    logger.save_log()
    
    if args.eval_cc:
        args.main_path = f'{args.ckpt_dir}/eval_cc'
        os.makedirs(args.main_path, exist_ok=True)
        logger = utils.Logger(
            log_dir=args.main_path,
            exp_name=args.main_path.replace('/', '_'),
            exp_suffix='',
            write_textfile=True,
            use_wandb=False,
            wandb_project_name=args.wandb_project_name,
            entity=args.wandb_entity
        )
        logger.update_config(args, is_args=True)

        total_top1, total_top5, total_objs = infer_common_corruption(args, model, criterion)
        logger.write_log_nohead({
            'valid/cc/loss': total_objs,
            'valid/cc/top1': total_top1,
            'valid/cc/top5': total_top5,
        }, step=0)

    logger.save_log()
    del model
        

def infer(args, epoch, valid_queue, model, criterion):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    adv_objs = utils.AvgrageMeter()
    adv_top1 = utils.AvgrageMeter()
    adv_top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, requires_grad=False).cuda(non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)
        
        if (epoch+1) % 10 == 0 or args.eval_only:
            if args.eval_only:
                if args.eval_attack_type == 'pgd':
                    adv_input = pgd_attack(args, model, input, target, step_size=0.003, epsilon=0.031, perturb_steps=args.num_steps, distance='l_inf')
                elif args.eval_attack_type == 'cw':
                    atk = torchattacks.CW(model)
                    adv_input = atk(input, target)
                elif args.eval_attack_type == 'deepfool':
                    atk = torchattacks.DeepFool(model)
                    adv_input = atk(input, target)
                elif args.eval_attack_type == 'spsa':
                    atk = torchattacks.SPSA(model)
                    adv_input = atk(input, target)
                elif args.eval_attack_type == 'lgv':
                    atk = torchattacks.LGV(model, valid_queue, attack_class=torchattacks.BIM, lr=0.05, epochs=2, nb_models_epoch=4, wd=1e-4, n_grad=1, eps=4/255, alpha=4/255/10, steps=10, verbose=True)
                    atk.collect_models()
                    atk.save_models(f'{args.save_dir}/{args.eval_attack_type}')
                    adv_input = atk(input, target)
                elif args.eval_attack_type == 'autoattack':
                    from autoattack import AutoAttack
                    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard')
                    adv_input = adversary.run_standard_evaluation(input, target, bs=input.size(0))
            else:
                adv_input = pgd_attack(args, model, input, target, step_size=0.003, epsilon=0.031, perturb_steps=20, distance='l_inf')
        
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                print(f'valid step: {step}, loss: {objs.avg:.3f}, top1: {top1.avg:.3f}, top5: {top5.avg:.3f}')
            
            if (epoch+1) % 10 == 0  or args.eval_only:
                logits = model(adv_input)
                adv_loss = criterion(logits, target)

                adv_prec1, adv_prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                adv_objs.update(adv_loss.data.item(), n)
                adv_top1.update(adv_prec1.data.item(), n)
                adv_top5.update(adv_prec5.data.item(), n)

                if step % args.report_freq == 0:
                    print(f'adv valid step: {step}, loss: {adv_objs.avg:.3f}, top1: {adv_top1.avg:.3f}, top5: {adv_top5.avg:.3f}')

    return top1.avg, top5.avg, objs.avg, adv_top1.avg, adv_top5.avg, adv_objs.avg


def infer_single_corruption(args, model, criterion, distortion):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.eval()

    if args.severity is not None:
        severity = [int(s) for s in args.severity.split(' ')]
    else:
        severity = None
    
    _, ds = get_dataloaders(
        args.batch_size, args.batch_size, args.dataset, args.num_data_workers, 
        normalize=True, return_ds=True, corruption=distortion, severity=severity, datadir=args.data
    )
    
    cc_data_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_data_workers, pin_memory=True
    )
    
    with torch.no_grad():
        for i, (input, target) in enumerate(cc_data_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(input, volatile=True).cuda()
            target_var = torch.autograd.Variable(target.long(), volatile=True).cuda()
            output = model(input_var).cuda()
            loss = criterion(output, target_var)

            prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

    return top1.avg, top5.avg, objs.avg


def infer_common_corruption(args, model, criterion):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    top1_list, top5_list, obj_list = [], [], []
    for d in DISTORTIONS:
        top1, top5, obj = infer_single_corruption(args, model, criterion, d)
        top1_list.append(top1)
        top5_list.append(top5)
        obj_list.append(obj)

    total_top1 = sum(top1_list)/len(top1_list)
    total_top5 = sum(top5_list)/len(top5_list)
    total_objs = sum(obj_list)/len(obj_list)
    return total_top1, total_top5, total_objs


def train(args, best_net, proxy_name=None):

    args.main_path = f'{args.save_dir}'
    args.main_path += f'/e-{args.epochs}-lr-{args.learning_rate}-wd-{args.weight_decay}'
    args.main_path += f'/net-{args.arch_type}'
    print(f'==> main_path: {args.main_path}')

    args.save = os.path.join(args.main_path, 'checkpoint')
    os.makedirs(args.main_path, exist_ok=True)
    
    # set logger
    print('==> Create Logger..')
    logger = utils.Logger(
        log_dir=args.main_path,
        exp_name=args.main_path.replace('/', '_'),
        exp_suffix='',
        write_textfile=True,
        use_wandb=False,
        wandb_project_name=args.wandb_project_name,
        entity=args.wandb_entity
    )
    logger.update_config(args, is_args=True)
    
    num_class = get_num_classes(args)
    
    # Find the architecture in the sampled lists
    if args.search_space == 'darts':
        
        genotype = Genotype(
            normal=best_net["normal"], normal_concat=best_net["normal_concat"],
            reduce=best_net["reduction"], reduce_concat=best_net["reduction_concat"],
        )
        model = Network(
            args.init_channels, num_class, args.layers, args.auxiliary,
            args.drop_path_prob, genotype
        )

    elif args.search_space == 'nasbench201':

        from nas_201_api import NASBench201API as API
        api = API(args.api_loc)
        arch_str = api[best_net]
        model = nasbench2.get_model_from_arch_str(arch_str, num_class)
        
    else:
        NotImplementedError()
        
    # Model & Optimizer setting
    model = model.cuda()            
    model_size = utils.count_parameters_in_MB(model)
    print(f"param size = {model_size}MB", )

    if args.dataset == 'ImageNet16-120':
        criterion = CrossEntropyLabelSmooth(num_class, args.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    if args.dataset == 'ImageNet16-120' and args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    else:
        lr_scheduler = None

    # DATASET setting
    train_queue, valid_queue = get_dataloaders(
        args.batch_size, args.batch_size, args.dataset, args.num_data_workers, 
        normalize=(True if args.adv_loss == '' else False), datadir=args.data
    )

    best_acc, best_adv_acc = 0.0, 0.0
    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch, args.learning_rate, dataset=args.dataset, lr_scheduler=lr_scheduler)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc1, train_acc5, train_obj, train_acc1_adv, train_acc5_adv, train_obj_adv = train_single(args, train_queue, model, criterion, optimizer, lr_scheduler)
        print(f'epoch: {epoch}, train acc: {train_acc1:.3f}, train adv acc: {train_acc1_adv:.3f}')

        valid_acc1, valid_acc5, valid_obj, valid_adv_acc1, valid_adv_acc5, valid_adv_obj = infer(args, epoch, valid_queue, model, criterion)
        print(f'epoch: {epoch}, valid acc: {valid_acc1:.3f}, valid adv acc: {valid_adv_acc1:.3f}')
        
        if valid_acc1 > best_acc:
            best_acc = valid_acc1
            utils.save_model({
                'epoch': epoch +1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args.save, model_name=f'best.pt')
            print(f'epoch: {epoch}, valid acc: {valid_acc1:.3f}, best acc: {best_acc:.3f}')

        if valid_adv_acc1 > best_adv_acc:
            best_adv_acc = valid_adv_acc1
            utils.save_model({
                'epoch': epoch +1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args.save, model_name=f'best_adv.pt')
            print(f'epoch: {epoch}, valid adv acc: {valid_adv_acc1:.3f}, best adv acc: {best_adv_acc:.3f}')

        logger.write_log_nohead({
            'epoch': epoch+1,
            'train/clean/loss': train_obj,
            'train/clean/top1': train_acc1,
            'train/clean/top5': train_acc5,
            'train/adv/loss': train_obj_adv,
            'train/adv/top1': train_acc1_adv,
            'train/adv/top5': train_acc5_adv,
            'valid/clean/loss': valid_obj,
            'valid/clean/top1': valid_acc1,
            'valid/clean/top5': valid_acc5,
            'valid/adv/loss': valid_adv_obj,
            'valid/adv/top1': valid_adv_acc1,
            'valid/adv/top5': valid_adv_acc5,
            'valid/adv/best': best_adv_acc,
            'valid/clean/best': best_acc
        }, step=epoch+1)
        
        utils.save_model({
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, args.save, model_name=f'last.pt')
        
        if (epoch+1) % 50 == 0:
            utils.save_model({
                'epoch': epoch + 1, 
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args.save, model_name=f'epoch_{epoch+1}.pt')

    logger.save_log()
    

def train_single(args, train_queue, model, criterion, optimizer, lr_scheduler):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    objs_adv = utils.AvgrageMeter()
    top1_adv = utils.AvgrageMeter()
    top5_adv = utils.AvgrageMeter()

    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda(non_blocking=True)
        target = Variable(target).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(input)
        
        if args.adv_loss == 'pgd':
            loss, logits_adv = madry_loss(
                args,
                model,
                input, 
                target, 
                optimizer,
                step_size = args.step_size,
                epsilon = args.epsilon, 
                perturb_steps = args.num_steps,
                return_logits=True
            )
            
        elif args.adv_loss == 'fgsm':
            loss, logits_adv = madry_loss(
                args,
                model,
                input, 
                target, 
                optimizer,
                step_size = 10./255.,
                epsilon = 8./255., 
                perturb_steps = 1,
                return_logits=True
            )
        
        else:
            loss = criterion(logits, target) 
            
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        
        if args.adv_loss in ['pgd', 'fgsm']:
            prec1_adv, prec5_adv = utils.accuracy(logits_adv, target, topk=(1, 5))
            objs_adv.update(loss.data.item(), n)
            top1_adv.update(prec1_adv.data.item(), n)
            top5_adv.update(prec5_adv.data.item(), n)
            if step % args.report_freq == 0:
                print(f'train step: {step}, loss: {objs.avg:.3f}, top1: {top1.avg:.3f}, top5: {top5.avg:.3f}, loss_adv: {objs_adv.avg:.3f}, top1_adv: {top1_adv.avg:.3f}, top5_adv: {top5_adv.avg:.3f}')
        else:
            if step % args.report_freq == 0:
                print(f'train step: {step}, loss: {objs.avg:.3f}, top1: {top1.avg:.3f}, top5: {top5.avg:.3f}')

    return top1.avg, top5.avg, objs.avg, top1_adv.avg, top5_adv.avg, objs_adv.avg


def main(args):
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    assert args.arch_type is not None
    if args.arch_type == 'advrush':
        archs = ADV_RUSH
    elif args.arch_type == 'drnas_cifar':
        archs = DRNAS_CIFAR
    elif args.arch_type == 'drnas_imagenet':
        archs = DRNAS_IMAGENET
    elif args.arch_type == 'pcdarts_cifar':
        archs = PCDARTS_CIFAR
    elif args.arch_type == 'pcdarts_imagenet':
        archs = PCDARTS_IMAGENET
    elif args.arch_type == 'croze_cifar10':
        archs = CROZE_CIFAR10
    elif args.arch_type == 'croze_cifar100':
        archs = CROZE_CIFAR100
    elif args.arch_type == 'croze_imagenet':
        archs = CROZE_IMAGENET

    if args.eval_only:
        test(args, archs)
    else:
        train(args, archs)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("train")
    parser.add_argument('--search_space', type=str, default='darts', help='search space (darts/nasbench201)')
    parser.add_argument('--api_loc', type=str, default='', help='path to API')
    parser.add_argument('--arch_type', type=str, default=None)

    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--save_dir', type=str, default='./exp')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--data', type=str, default='', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size') #128
    parser.add_argument('--lr_scheduler', type=str, default='linear', help=['linear', 'cosine'])
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    parser.add_argument('--epsilon', type=float, default=8./255., help='perturbation')
    parser.add_argument('--num_steps', type=int, default=7, help='perturb number of steps')
    parser.add_argument('--step_size', type=float, default=2./255., help='perturb step size')
    parser.add_argument('--adv_loss', type=str, default='', help=['', 'pgd', 'fgsm'])
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--wandb_project_name', type=str)
    parser.add_argument('--wandb_entity', type=str)

    # eval args
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_attack_type', type=str, default='pgd')
    parser.add_argument('--eval_cc', action='store_true')
    parser.add_argument('--severity', type=int, default=None)
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)


