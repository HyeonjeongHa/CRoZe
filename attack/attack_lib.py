import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from torch.autograd import Variable


def pgd_attack(
    args, model, x_natural, y, step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    distance='l_inf',
):

    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    if perturb_steps == 1:
        x_adv = x_natural.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, x_natural.shape)).float().cuda()
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_ce = criterion_ce(logits, y).mean()
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    return x_adv.detach()


def madry_loss(
    args, model,
    x_natural,
    y,
    optimizer,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    distance='l_inf',
    return_logits=False
):

    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, x_natural.shape)).float().cuda()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_ce = criterion_ce(logits, y).mean()
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    if return_logits:
        return loss, logits
    return loss