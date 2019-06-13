from core.config import config
import torch
from PIL import Image
import cv2
import torchvision.transforms as T
import numpy as np


class RandomColor(object):
    def __init__(self):
        self.color_jitter = T.ColorJitter(brightness=config.train.augment.brightness,
                                     contrast = config.train.augment.contrast,
                                     saturation = config.train.augment.saturation)
    def __call__(self, batch):
        image = batch['image']
        image = self.color_jitter(image)
        batch['image'] = image
        return batch

class RandomScale(object):
    def __init__(self, min_size, is_depth):
        self.min_size = min_size
        self.is_depth = is_depth
    def __call__(self, batch):
        image = np.array(batch['image'])[:, :, -1::-1].astype(np.float32)
        gt = batch['gt']
        size = np.random.choice(self.min_size)
        scale = np.float32(size) / np.float32(np.min(image.shape[:2]))
        image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if self.is_depth:
            gt /= scale
        else:
            gt *= scale
        batch['image'] = image
        batch['gt'] = gt
        return batch
class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, batch):
        flip = np.random.rand()<self.prob
        if flip:
            batch['image'] = np.array(batch['image'][:, -1::-1])
            batch['gt'] = np.array(batch['gt'][:, -1::-1])
        return batch

class RandomCrop(object):
    def __init__(self):
        self.crop_size = config.train.augment.crop_size

    def __call__(self, batch):
        image = batch['image']
        gt = batch['gt']
        h, w = image.shape[:2]
        # pad if needed
        pad_h = [0, 0]
        pad_w = [0, 0]
        if h < self.crop_size:
            pad_h[0] = (self.crop_size - h) / 2
            pad_h[1] = self.crop_size - h - pad_h[0]
        if w < self.crop_size:
            pad_w[0] = (self.crop_size -w) / 2
            pad_w[1] = self.crop_size - w - pad_w[0]
        image = np.pad(image, [pad_h, pad_w, [0, 0]], mode='constant')
        gt = np.pad(gt, [pad_h, pad_w], mode='constant', constant_values=-1.0)
        h, w = image.shape[:2]
        y0 = np.random.randint(0, h - self.crop_size + 1)
        x0 = np.random.randint(0, w - self.crop_size + 1)
        image = image[y0:y0+self.crop_size, x0:x0+self.crop_size]
        gt = gt[y0:y0+self.crop_size, x0:x0+self.crop_size]
        batch['image'] = image
        batch['gt'] = gt
        return batch
        

class Normalize(object):
    def __init__(self):
        self.pixel_mean = config.train.augment.pixel_mean.reshape((1, 1, -1))
    def __call__(self, batch):
        batch['image'] -= self.pixel_mean
        return batch

class ToTensor(object):
    def __init__(self):
        self.gpu = config.gpu[0]
    def __call__(self, batch):
        image = np.transpose(batch['image'], [2, 0, 1])
        gt = batch['gt'][None, ...]
        #batch['image'] = torch.tensor(image).pin_memory().to(self.gpu, non_blocking=True)
        #batch['gt'] = torch.tensor(gt).pin_memory().to(self.gpu, non_blocking=True)
        batch['image'] = image
        batch['gt'] = gt
        return batch

class Transform(object):
    def __init__(self, phase, is_depth=False):
        if phase == 'train':
            self.transforms = T.Compose([RandomColor(), RandomScale(config.train.augment.random_resize, is_depth), 
                               RandomCrop(), RandomFlip(), Normalize(), ToTensor()]
)
        else:
            self.transforms = T.Compose([RandomScale(config.test.augment.min_size, is_depth), Normalize(), ToTensor()])

    def __call__(self, batch):
        batch = self.transforms(batch)
        return batch


        
