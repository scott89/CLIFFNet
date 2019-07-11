from easydict import EasyDict as edict
import numpy as np
import os
#import yaml


config = edict()

config.dataset = edict()
config.dataset.name = 'NYU'
config.dataset.data_path = '/media/4TB/Research/DataSet/NYU2/hu_data/data/'
config.dataset.train_data_file = 'train_list.p'
config.dataset.val_data_file = 'test_list.p'
config.dataset.scale = 0.5

config.network = edict()
config.network.backbone_fix_bn = False
config.network.backbone_with_dilation = False
config.network.backbone_freeze_at = 0

config.train = edict()
config.train.augment = edict()
config.train.augment.brightness = 0.4
config.train.augment.contrast = 0.4
config.train.augment.saturation = 0.4
config.train.augment.crop_size = 427
config.train.augment.random_resize = [459, 427, 400, 368]
config.train.augment.pixel_mean = np.array((102.9801, 115.9465, 122.7717,))

config.train.batch_size = 8
config.train.lr = 1e-4
config.train.lr_decay_rate = 0.1
config.train.warmup_epoch = 0
config.train.warmup_lr = 1e-3
config.train.momentum = 0.9
config.train.weight_decay = 1e-6
config.train.max_epoch = 500
config.train.lr_decay_epoch = 18
config.train.display_iter = 20
config.train.summary_iter = 100
config.train.output_path = 'models/nyu_v1.2.5/'

config.train.resume = False
config.train.snapshot = 'models/nyu_v1.2/epoch-17.pth'
config.train.pretrained = False
config.train.pretrained_backbone = 'models/resnet-50-caffe.pth'

config.test = edict()
config.test.batch_size = 1
config.test.augment = edict()
config.test.augment.min_size = [427]

config.gpu_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
config.gpu = ['cuda:%s'%i for i,j in enumerate(config.gpu_id.split(','))]
