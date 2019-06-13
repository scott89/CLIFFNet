import torch
from core.config import config


class BaseOptimizer():
    def decay_lr(self, factor=0.1):
        for p in self.param_groups:
            p['lr'] *= factor
        self.defaults['lr'] *= factor

    def set_lr(self, lr):
        for p in self.param_groups:
            p['lr'] = lr
        self.defaults['lr'] = lr

class SGD(torch.optim.SGD, BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

class Adam(torch.optim.Adam, BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)
