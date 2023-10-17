import os
import numpy as np
import torch
from torch.autograd import Variable
import wandb


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
  
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
  

def drop_path(x, drop_prob):
  
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_model(checkpoint, save_path, model_name=None):
    os.makedirs(save_path, exist_ok=True)

    if model_name is None:
        model_name = 'checkpoint.pt'
    torch.save(checkpoint, os.path.join(save_path, model_name))
    

class Logger:
    def __init__(
        self,
        exp_name,
        log_dir=None,
        exp_suffix="",
        write_textfile=True,
        use_wandb=False,
        wandb_project_name=None,
        entity='sh'
    ):

        self.log_dir = log_dir
        self.write_textfile = write_textfile
        self.use_wandb = use_wandb

        self.logs_for_save = {}
        self.logs = {}

        if self.write_textfile:
            self.f = open(os.path.join(log_dir, 'logs.txt'), 'w')

        if self.use_wandb:
            exp_suffix = "_".join(exp_suffix.split("/")[:-1])
            self.run = wandb.init(
                config=wandb.config,
                entity=entity,
                project=wandb_project_name, 
                name=exp_name + "_" + exp_suffix, 
                group=exp_name.split('net-')[0],
                reinit=True)
            
            
    def update_config(self, v, is_args=False):
      
        if is_args:
            self.logs_for_save.update({'args': v})
        else:
            self.logs_for_save.update(v)
        if self.use_wandb:
            wandb.config.update(v, allow_val_change=True)


    def write_log_nohead(self, element, step):
      
        log_str = f"{step} | "
        log_dict = {}
    
        logs_for_save = self.logs_for_save
        f = self.f
        
        for key, val in element.items():
            if not key in logs_for_save:
                logs_for_save[key] =  []
            logs_for_save[key].append(val)
            log_str += f'{key} {val} | '
            log_dict[f'{key}'] = val
        
        if self.write_textfile:
            f.write(log_str+'\n')
            f.flush()

        if self.use_wandb:
            wandb.log(log_dict, step=step)
      
            
    def write_log(self, element, step, img_dict=None, tbl_dict=None):
      
        log_str = f"{step} | "
        log_dict = {}
        
        logs_for_save = self.logs_for_save
        logs = self.logs
        f = self.f
          
        for head, keys  in element.items():
            for k in keys:
                v = logs[k].avg
                if not k in logs_for_save:
                    logs_for_save[k] = []
                logs_for_save[k].append(v)
                log_str += f'{k} {v}| '
                log_dict[f'{head}/{k}'] = v

        if self.write_textfile:
            f.write(log_str+'\n')
            f.flush()
        
        if img_dict is not None:
            log_dict.update(img_dict)
        
        if tbl_dict is not None:
            log_dict.update(tbl_dict)
            
        if self.use_wandb:
            wandb.log(log_dict, step=step)


    def save_log(self, name=None):
        torch.save(self.logs_for_save, os.path.join(self.log_dir, 'logs.pt'))
    

    def update(self, key, v, n=1):
      
        logs = self.logs 
        if not key in logs:
            logs[key] = AverageMeter()
        logs[key].update(v, n)
    

    def reset(self, keys=None, except_keys=[]):
      
        logs = self.logs
        if keys is not None:
            if isinstance(keys, list):
                for key in keys:
                    logs[key] =  AverageMeter()
            else:
                logs[keys] = AverageMeter()
        else:
            for key in logs.keys():
                if not key in except_keys:
                    logs[key] = AverageMeter()


    def avg(self, keys=None, except_keys=[]):
      
        logs = self.logs
        if keys is not None:
            if isinstance(keys, list):
                return {key: logs[key].avg for key in keys if key in logs.keys()}
            else:
                return logs[keys].avg
        else:
            avg_dict = {}
            for key in logs.keys():
                if not key in except_keys:
                    avg_dict[key] =  logs[key].avg
            return avg_dict 


