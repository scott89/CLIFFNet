from torch import nn
from network.resnet import get_params
from core.config import config
from core.optimizer import SGD, Adam


def get_param_lr(net):
    bn_names = []
    bn_params = []
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_params.extend([module.weight, module.bias])
            bn_names.append(name+'.weight')
            bn_names.append(name+'.bias')
    params = [
        {'params': bn_params, 'lr_mul': 1.0, 'weight_decay':0},
        {'params': get_params(net, ['res_backone'], ['weight'], bn_names), 'lr_mul': 1.0},
        {'params': get_params(net, ['fcn'], 'weight', bn_names), 'lr_mul': 1.0},
        {'params': get_params(net, ['fcn'], 'bias', bn_names), 'lr': 2 * config.train.lr, 'weight_decay': 0.0, 'lr_mul': 2.0},
        {'params': get_params(net, ['fpn'], 'weight', bn_names), 'lr_mul': 1.0},
        {'params': get_params(net, ['fpn'], 'bias', bn_names), 'lr': 2 * config.train.lr, 'weight_decay': 0.0, 'lr_mul': 2.0}
    ]
    return params


def build_optimizer(net):
    params = get_param_lr(net.module)
    #opt = SGD(params, lr=config.train.lr, momentum=config.train.momentum, weight_decay=config.train.weight_decay)
    opt = Adam(params, lr=config.train.lr, weight_decay=config.train.weight_decay)
    return opt
