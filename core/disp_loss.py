import torch
import numpy as np

def l1_loss(gt, pre, mask=None):
    if mask is None:
        mask = gt
    pre = pre[mask>=0]
    gt = gt[mask>=0]
    loss = torch.mean(torch.abs(gt - pre))
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
    

def log_loss(gt, pre, mask=None, a=1.0, b=1/2.0, thr=0):
    if mask is None:
        mask = gt
    pre = pre[mask>=0]
    gt = gt[mask>=0]
    diff = torch.abs(gt - pre)
    loss = torch.log(a*(diff+b))
    valid_flag = diff>thr
    loss = loss[valid_flag]
    loss = torch.mean(loss)
    return loss

def compute_metrics(gt, pre, delta):
    gt = gt.detach().cpu().numpy()
    pre = pre.detach().cpu().numpy()
    pre = np.clip(pre, 0, 10)
    pre = pre[gt>=0]
    gt = gt[gt>=0]
    abs_diff = np.abs(gt - pre)
    rms = (np.mean(abs_diff**2))**0.5
    gt[gt==0] += 0.001
    pre[pre==0] += 0.001
    rel = np.mean(abs_diff / gt)
    rms_log10 = (np.mean(np.abs(np.log(pre) - np.log(gt))**2))**0.5
    r1 = pre / gt
    r2 = gt / pre
    r = np.maximum(r1, r2)
    precision = []
    for d in delta:
        precision.append(np.mean(np.float32(r < d)))
    return rms, rel, rms_log10, precision

