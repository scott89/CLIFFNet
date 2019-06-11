import torch
from core.config import config


class SGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def decay_lr(self, factor=0.1):
        for p in self.param_groups:
            p['lr'] *= factor
        self.defaults['lr'] *= factor

