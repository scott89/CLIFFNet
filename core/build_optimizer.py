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
    res_backbone_w = get_params(net, ['res_backbone'], ['weight'], bn_names)
    res_backbone_b = get_params(net, ['res_backbone'], ['bias'], bn_names)
    fpn_w = get_params(net, ['fpn'], ['weight'], bn_names)
    fpn_b = get_params(net, ['fpn'], ['bias'], bn_names)
    fcn_w = get_params(net, ['fcn'], ['weight'], bn_names)
    fcn_b = get_params(net, ['fcn'], ['bias'], bn_names)
    params = [
        {'params': bn_params, 'params_names':bn_names, 'lr': 1.0, 'weight_decay':0},
        {'params': res_backbone_w[0], 'params_names': res_backbone_w[1], 'lr': 1.0},
        {'params': res_backbone_b[0], 'params_names': res_backbone_b[1], 'lr': 1.0, 'weight_decay': 0.0},
        {'params': fpn_w[0], 'params_names': fpn_w[1], 'lr': 1.0},
        {'params': fpn_b[0], 'params_names': fpn_b[1], 'lr': 1.0, 'weight_decay': 0.0},
        {'params': fcn_w[0], 'params_names': fcn_w[1], 'lr': 1.0},
        {'params': fcn_b[0], 'params_names': fcn_b[1], 'lr': 1.0, 'weight_decay': 0.0},
    ]
    return params


def build_optimizer(net):
    params = get_param_lr(net.module)
    opt = SGD(params, lr=1.0, momentum=config.train.momentum, weight_decay=config.train.weight_decay)
    #opt = Adam(params, lr=config.train.lr, weight_decay=config.train.weight_decay)
    return opt
