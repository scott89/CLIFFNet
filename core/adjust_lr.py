import torch
from os.path import join
from core.config import config
from core.build_optimizer import build_optimizer



def get_step_index(it, decay_iterations):
    for idx, decay_iteration in enumerate(decay_iterations):
        if it < decay_iteration:
            return idx
    return len(decay_iterations)



def adjust_lr(base_lr, it, decay_iterations, optimizer):
    if it <= config.train.warmup_it:
        alpha = 1.0 * it / config.train.warmup_it
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)
    if it in decay_iterations:
        try:
            for k in optimizer.state_dict()['state'].keys():
                optimizer.state_dict()['state'][k]['momentum_buffer'].div_(10)
        except:
            pass
    return base_lr * (0.1 ** get_step_index(it, decay_iterations))
