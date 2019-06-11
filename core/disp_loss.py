import torch

def l1_loss(gt, pre):
    valid_mask = (gt>=0).float()
    loss = torch.sum(torch.abs(gt - pre) * valid_mask) / (torch.sum(valid_mask) + 1e-6)
    return loss
