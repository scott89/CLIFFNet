import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
import numpy as np
import torchvision.transforms as T
from core.config import config
from core.data_transform import Transform

class Cityscape(Dataset):
    def __init__(self, data_file, phase):
        self.data_path = config.dataset.data_path
        self.phase = phase
        self.im_list = np.load(os.path.join(self.data_path, data_file))
        self.transform = Transform(phase)

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        im_name, gt_name = self.im_list[idx]
        image = Image.open(os.path.join(self.data_path, im_name))
        gt = np.array(Image.open(os.path.join(self.data_path, gt_name))).astype(np.float32)
        gt = (gt - 1) / 256.0
        batch = {'image': image, 'gt': gt}
        batch = self.transform(batch)
        return batch

            



