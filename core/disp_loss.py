import torch

def l1_loss(gt, pre):
    valid_mask = (gt>=0).float()
    loss = torch.sum(torch.abs(gt - pre) * valid_mask) / (torch.sum(valid_mask) + 1e-6)
    return loss

def l2_loss(gt, pre):
    valid_mask = (gt>=0).float()
    loss = torch.sum(((gt - pre) * valid_mask)**2) / (torch.sum(valid_mask) + 1e-6)
    return loss


def huber_loss(gt, pre, delta=0.5):
    pre = pre[gt>=0]
    gt = gt[gt>=0]
    abs_diff = torch.abs(gt - pre)
    l1_loss = delta * (abs_diff - 0.5*delta)
    l2_loss = 0.5 * abs_diff **2
    flag = (abs_diff <= delta).float()
    loss = flag * l2_loss + (1-flag) * l1_loss
    loss = torch.mean(loss)
    return loss
    
