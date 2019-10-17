from __future__ import division
from core.config import config
import torch
from PIL import Image
import cv2
import torchvision.transforms as T
import numpy as np
import random


class RandomColor(object):
    def __init__(self, brightness, contrast, saturation):
        self.color_jitter = T.ColorJitter(brightness=brightness,
                                     contrast = contrast,
                                     saturation = saturation)
    def __call__(self, batch):
        image = batch['data']
        image = self.color_jitter(image)
        batch['data'] = image
        return batch


class RandomRotate(object):
    def __init__(self, max_degree):
        self.max_degree = max_degree
    def __call__(self, batch):
        image = batch['data']
        gt = batch['gt']
        degree = random.uniform(-self.max_degree, self.max_degree)
        image = T.functional.rotate(image, degree, resample=Image.BILINEAR)
        gt = T.functional.rotate(gt, degree, resample=Image.BILINEAR)
        batch['data'] = image
        batch['gt'] = gt
        return batch

class ToArray(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        image = np.array(batch['data']).astype(np.float32)
        gt = np.array(batch['gt']).astype(np.float32) / 255.0 * 10.0
        if len(image.shape) == 3:
            image = image[:, :, -1::-1]       
        else:
            image = np.tile(image[:,:,None], [1,1,3])

        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        batch['data'] = image
        batch['gt'] = gt
        return batch




class CenterCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, batch):
        image = batch['data']
        gt = batch['gt']
        h, w = gt.shape
        y1 = int(round(h - self.crop_size[0]))
        x1 = int(round(w - self.crop_size[1]))
        image = image[y1:y1+self.crop_size[0], x1:x1+self.crop_size[1]]
        gt = gt[y1:y1+self.crop_size[0], x1:x1+self.crop_size[1]]
        batch['data'] = image
        batch['gt'] = gt
        return batch



class Resize(object):
    def __init__(self, target_sizes, max_size, canonical_size):
        self.max_size = max_size
        self.target_sizes = target_sizes
        self.canonical_size = canonical_size

    def __call__(self, batch):
        image = batch['data']
        gt = batch['gt']
        size_id = random.randint(0, len(self.target_sizes)-1)
        target_size = self.target_sizes[size_id]
        im_size_min = min(image.shape[:2])
        im_size_max = max(image.shape[:2])
        scale = target_size / im_size_min
        if scale * im_size_max > self.max_size:
            scale = self.max_size / im_size_max
        depth_scale = self.canonical_size / (scale * im_size_min)
        image = cv2.resize(image, None, None, scale, scale, interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, None, None, scale, scale, interpolation=cv2.INTER_LINEAR)
        gt *= depth_scale
        batch['data'] = image
        batch['gt'] = gt
        batch['im_info'] = list(image.shape[:2]) + [scale]
        return batch

class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, batch):
        flip = random.uniform(0, 1)<self.prob
        if flip:
            batch['data'] = np.array(batch['data'][:, -1::-1])
            batch['gt'] = np.array(batch['gt'][:, -1::-1])
        return batch

class Normalize(object):
    def __init__(self, pixel_mean):
        self.pixel_mean = pixel_mean.reshape((1, 1, -1))
    def __call__(self, batch):
        batch['data'] -= self.pixel_mean
        return batch

class HWC2CHW(object):
    def __call__(self, batch):
        batch['data'] = np.transpose(batch['data'], [2, 0, 1])
        batch['gt'] = batch['gt'][None, ...]
        return batch

class Transform(object):
    def __init__(self, phase, is_depth=False):
        if phase == 'train':
            self.transforms = T.Compose([RandomColor(config.train.augment.brightness, 
                                                     config.train.augment.contrast, 
                                                     config.train.augment.saturation), 
                                         RandomRotate(config.train.augment.rotation),
                                         ToArray(), CenterCrop(config.dataset.crop_size), 
                                         Resize(config.train.augment.min_size, 
                                                config.train.augment.max_size,
                                                config.train.augment.canonical_size), 
                                         RandomFlip(), Normalize(config.network.pixel_mean), HWC2CHW()])
        else:
            self.transforms = T.Compose([ToArray(), CenterCrop(config.dataset.crop_size), 
                                         Resize(config.test.augment.min_size, 
                                                config.test.augment.max_size,
                                                config.test.augment.canonical_size), 
                                         Normalize(config.network.pixel_mean), HWC2CHW()])

    def __call__(self, batch):
        batch = self.transforms(batch)
        return batch


        
