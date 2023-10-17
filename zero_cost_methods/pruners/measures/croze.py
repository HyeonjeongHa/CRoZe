import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import copy
from . import measure
from ..p_utils import adj_weights, get_layer_metric_array_adv_feats


def fgsm_attack(net, image, target, epsilon):
    perturbed_image = image.detach().clone()
    perturbed_image.requires_grad = True
    net.zero_grad()

    logits = net(perturbed_image)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    
    sign_data_grad = perturbed_image.grad.sign_()
    perturbed_image = perturbed_image - epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


@measure('croze', bn=False, mode='param')
def compute_synflow_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None, search_space='nasbench201'):

    device = inputs.device
    origin_inputs, origin_outputs = inputs, targets
    
    cos_loss = nn.CosineSimilarity(dim=0)
    ce_loss = nn.CrossEntropyLoss()
    
    #convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    #convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])


    advnet = copy.deepcopy(net)
    
    # keep signs of all params
    signs = linearize(net)
    adv_signs = linearize(advnet)
    
    # Compute gradients with input of 1s 
    net.zero_grad()
    net.double()
    advnet.double()
    
    output, feats = net.forward(origin_inputs.double(), return_activated_feats=True)
    output.retain_grad()

    advnet = adj_weights(advnet, origin_inputs.double(), origin_outputs, 2.0, loss_maximize=True)
    advinput = fgsm_attack(advnet, origin_inputs.double(), origin_outputs, 0.01)

    advnet.train()
    adv_outputs, adv_feats = advnet.forward(advinput.detach(), return_activated_feats=True)
    adv_outputs.retain_grad()
    
    loss = ce_loss(output, origin_outputs) + ce_loss(adv_outputs, origin_outputs)
    loss.backward() 

    def croze(layer, layer_adv, feat, feat_adv):
        if layer.weight.grad is not None:
            w_sim = (1+cos_loss(layer_adv.weight, layer.weight)).sum()
            sim = (torch.abs(cos_loss(layer_adv.weight.grad, layer.weight.grad))).sum()
            feat_sim = (1+cos_loss(feat_adv, feat)).sum()
            return torch.abs(w_sim * sim * feat_sim)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array_adv_feats(net, advnet, feats, adv_feats, croze, mode, search_space) 

    # apply signs of all params
    nonlinearize(net, signs)
    nonlinearize(advnet, adv_signs)
    
    del feats, adv_feats
    del advnet
    
    return grads_abs
