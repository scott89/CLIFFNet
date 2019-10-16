import torch
from core.config import config


class BaseOptimizer():
    def decay_lr(self, factor=0.1):
        for p in self.param_groups:
            p['lr'] *= factor
        self.defaults['lr'] *= factor

    def set_lr(self, lr):
        for p in self.param_groups:
            p['lr'] = lr*p['lr_mul']
        self.defaults['lr'] = lr

class SGD(torch.optim.SGD, BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)
    def step(self, lr, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.new().resize_as_(p.data).zero_()
                        buf.mul_(momentum).add_(group['lr']*lr, d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(group['lr']*lr, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                    p.data.add_(-1, d_p)
                else:
                    p.data.add_(-group['lr']*lr, d_p)

        return loss

class Adam(torch.optim.Adam, BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)
