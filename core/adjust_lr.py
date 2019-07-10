import torch
from os.path import join
from core.config import config
from core.build_optimizer import build_optimizer

def adjust_lr(epoch, net, opt, best_epoch):
    if epoch < config.train.warmup_epoch:
        opt.set_lr(lr = config.train.warmup_lr)
    elif epoch == config.train.warmup_epoch:
        opt.set_lr(lr = config.train.lr)
    elif (epoch%config.train.lr_decay_epoch == 0) and (epoch != 0):
        ckpt = torch.load(
            join(config.train.output_path, 'epoch-%d.pth'%best_epoch), 
            map_location = config.gpu[0])
        net.load_state_dict(ckpt['model_state_dict'])
        opt = build_optimizer(net)
        #opt.decay_lr(factor = config.train.lr_decay_rate**(epoch/config.train.lr_decay_epoch))
        lr = config.train.lr * config.train.lr_decay_rate**(epoch/config.train.lr_decay_epoch)
        opt.set_lr(lr = lr)
    return net, opt

