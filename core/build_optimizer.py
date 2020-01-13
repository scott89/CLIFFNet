from torch import nn
from network.resnet import get_params
from core.config import config
from core.optimizer import SGD, Adam
import re


def get_param_lr(net):
    bn_backbone_names = []
    bn_backbone_params = []
    bn_dec_names = []
    bn_dec_params = []
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if re.search('res_backbone', name):
                bn_backbone_params.extend([module.weight, module.bias])
                bn_backbone_names.append(name+'.weight')
                bn_backbone_names.append(name+'.bias')
            else:
                bn_dec_params.extend([module.weight, module.bias])
                bn_dec_names.append(name+'.weight')
                bn_dec_names.append(name+'.bias')

    bn_names = bn_backbone_names + bn_dec_names
    res_backbone_w = get_params(net, ['res_backbone'], ['weight'], bn_names)
    res_backbone_b = get_params(net, ['res_backbone'], ['bias'], bn_names)
    fpn_w = get_params(net, ['fpn'], ['weight'], bn_names)
    fpn_b = get_params(net, ['fpn'], ['bias'], bn_names)
    fcn_w = get_params(net, ['fcn'], ['weight'], bn_names)
    fcn_b = get_params(net, ['fcn'], ['bias'], bn_names)
    params = [
        {'params': bn_backbone_params, 'params_names':bn_backbone_names, 'lr': 1.0, 'weight_decay':0},
        {'params': res_backbone_w[0], 'params_names': res_backbone_w[1], 'lr': 1.0},
        {'params': res_backbone_b[0], 'params_names': res_backbone_b[1], 'lr': 1.0, 'weight_decay': 0.0},
        {'params': bn_dec_params, 'params_names':bn_dec_names, 'lr': 10.0, 'weight_decay':0},
        {'params': fpn_w[0], 'params_names': fpn_w[1], 'lr': 10.0},
        {'params': fpn_b[0], 'params_names': fpn_b[1], 'lr': 10.0, 'weight_decay': 0.0},
        {'params': fcn_w[0], 'params_names': fcn_w[1], 'lr': 10.0},
        {'params': fcn_b[0], 'params_names': fcn_b[1], 'lr': 10.0, 'weight_decay': 0.0},
    ]
    return params


def build_optimizer(net):
    params = get_param_lr(net.module)
    opt = SGD(params, lr=1.0, momentum=config.train.momentum, weight_decay=config.train.weight_decay)
    #opt = Adam(params, lr=1.0, weight_decay=config.train.weight_decay)
    return opt
