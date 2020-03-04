import os
import numpy as np
import cv2
from core.config import config
from core.datasets.cityscape import Cityscape
from core.datasets.nyu import NYU
import torch
from torch.utils.data import DataLoader
from network.net import Net
from core.build_network import build_network
from core.disp_loss import compute_metrics


def normalize(x, gt=None):
    if gt is None:
        gt = x
    mask = gt >= 0
    x_min = gt[mask].min()
    x_max = gt.max()
    x = np.clip(x, x_min, x_max)
    x = (x - x_min) / (x_max - x_min)
    x = x * np.float32(mask)
    return x


dataset = eval(config.dataset.name)(config.test.data_file, 'val')
data_loader = DataLoader(dataset, 
                             batch_size=1, 
                             shuffle=False, 
                             num_workers=10, 
                             collate_fn=dataset.collate)
net = build_network()
ckpt = torch.load(config.test.snapshot, map_location = config.gpu[0])
net.load_state_dict(ckpt['model_state_dict'])

if config.test.save_res:
    if not os.path.isdir(config.test.save_path):
        os.makedirs(config.test.save_path)


rms, rel, sq_rel, rms_log10, p1, p2, p3 = 0, 0, 0, 0, 0, 0, 0
net.module.set_stage('eval')
for i, batch in enumerate(data_loader):
    image = batch['data'].pin_memory().to(config.gpu[0])
    gt = batch['gt'].pin_memory().to(config.gpu[0])
    with torch.no_grad():
        prediction = net(image)
        #prediction = torch.nn.functional.interpolate(prediction, gt.shape[2:], mode='bilinear')
    metrics = compute_metrics(gt, prediction, [1.25, 1.25**2, 1.25**3])
    rms += metrics[0]
    rel += metrics[1]
    sq_rel += metrics[2]
    rms_log10 += metrics[3]
    p1 += metrics[4][0]
    p2 += metrics[4][1]
    p3 += metrics[4][2]
    if config.test.save_res:
        gt = gt[0,0].cpu().numpy()
        pre = prediction[0,0].cpu().numpy()
        im = image[0].cpu().numpy()
        pre = 1.0-normalize(pre, gt)
        gt = 1.0-normalize(gt)
        gt = cv2.applyColorMap(np.uint8(gt*255), cv2.COLORMAP_INFERNO)
        pre = cv2.applyColorMap(np.uint8(pre*255), cv2.COLORMAP_INFERNO)
        im = np.transpose(im, [1,2,0])
        im += config.network.pixel_mean
        cv2.imwrite(config.test.save_path+'%04d-im.png'%i, im)
        cv2.imwrite(config.test.save_path+'%04d-gt.png'%i, gt)
        cv2.imwrite(config.test.save_path+'%04d-pre.png'%i, pre)
rms = (rms / len(data_loader))
rel /= len(data_loader)
sq_rel /= len(data_loader)
rms_log10 /= len(data_loader)
p1 /= len(data_loader)
p2 /= len(data_loader)
p3 /= len(data_loader)

epoch = 1
print("Epoch: %d, Val rms: %f, rel: %f, sq_rel: %f, rms_log10: %f, \
      p1: %f, p2: %f, p3: %f"%(epoch, rms, rel, sq_rel, rms_log10, p1, p2, p3))
