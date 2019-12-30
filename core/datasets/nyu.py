import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
import numpy as np
import torchvision.transforms as T
from core.config import config
from core.data_transform import Transform

class NYU(Dataset):
    def __init__(self, data_file, phase):
        self.data_path = config.dataset.data_path
        self.phase = phase
        self.im_list = np.load(os.path.join(self.data_path, data_file), allow_pickle=True)
        self.transform = Transform(phase, is_depth=True)

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        im_name, gt_name = self.im_list[idx][:2]
        image = Image.open(os.path.join(self.data_path, im_name))
        gt = Image.open(os.path.join(self.data_path, gt_name))
        batch = {'data': image, 'gt': gt}
        batch = self.transform(batch)
        #if self.phase == 'train':
        #    batch['gt'] = cv2.resize(batch['gt'][0], (batch['gt'].shape[2]/2, batch['gt'].shape[1]/2), interpolation=cv2.INTER_LINEAR)
        #    batch['gt'] = batch['gt'][None, ...]
        return batch

    def collate(self, batch):
        blob = dict()
        for key in batch[0]:
            if key == 'data':
                #blob['data'] = torch.from_numpy(self.im_list_to_blob([b['data'] for b in batch]))
                blob['data'] = torch.from_numpy(np.stack([b['data'] for b in batch], 0))
            elif key == 'gt':
                #blob['gt'] = torch.from_numpy(self.im_list_to_blob([b['gt'] for b in batch], 1, -1.0))
                blob['gt'] = torch.from_numpy(np.stack([b['gt'] for b in batch], 0))
            elif key == 'im_info':
                blob['im_info'] = np.array([b['im_info'] for b in batch])
            else:
                raise ValueError('Unknown batch key: %s'%key)
        return blob

    def im_list_to_blob(self, ims, num_channels=3, default_value=0.0):
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        stride = float(config.network.stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
        max_shape[2] = int(np.ceil(max_shape[2] / stride) * stride)

        num_images = len(ims)
        blob = default_value * np.ones((num_images, num_channels, max_shape[1], max_shape[2]), dtype=np.float32)
        for i in range(num_images):
            im = ims[i]             
            blob[i, :, 0:im.shape[1], 0:im.shape[2]] = im
        return blob


