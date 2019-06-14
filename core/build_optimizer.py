from network.resnet import get_params
from core.config import config
from core.optimizer import SGD, Adam


def get_param_lr(net):
    params = [
        {'params': get_params(net, ['res_backone'], ['weight']), 'lr_mul': 1.0},
        {'params': get_params(net, ['fcn'], 'weight'), 'lr_mul': 1.0},
        {'params': get_params(net, ['fcn'], 'bias'), 'lr': 2 * config.train.lr, 'weight_decay': 0.0, 'lr_mul': 2.0},
        {'params': get_params(net, ['fpn'], 'weight'), 'lr_mul': 1.0},
        {'params': get_params(net, ['fpn'], 'bias'), 'lr': 2 * config.train.lr, 'weight_decay': 0.0, 'lr_mul': 2.0}
    ]
    return params


def build_optimizer(net):
    params = get_param_lr(net.module)
    opt = SGD(params, lr=config.train.lr, momentum=config.train.momentum, weight_decay=config.train.weight_decay)
    #opt = Adam(params, lr=config.train.lr, weight_decay=config.train.weight_decay)
    return opt
