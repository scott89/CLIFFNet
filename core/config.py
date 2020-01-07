from easydict import EasyDict as edict
import numpy as np
import os
#import yaml


config = edict()

config.dataset = edict()
config.dataset.name = 'NYU'
config.dataset.data_path = '/media/4TB/Research/DataSet/NYU2/'
config.dataset.train_data_file = 'train_40k.p'
config.dataset.val_data_file = 'test.p'
config.dataset.crop_size = [456, 608]

config.network = edict()
config.network.backbone_fix_bn = False
config.network.backbone_with_dilation = False
config.network.backbone_freeze_at = 0
config.network.pixel_mean = np.array((102.9801, 115.9465, 122.7717,))
config.network.stride = 32

config.train = edict()
config.train.augment = edict()
config.train.augment.brightness = 0.4
config.train.augment.contrast = 0.4
config.train.augment.saturation = 0.4
config.train.augment.min_size = [228]
config.train.augment.canonical_size = 228
config.train.augment.max_size = [512]
config.train.augment.rotation = -1

config.train.batch_size = 8
config.train.lr = 1e-4
config.train.lr_decay_iterations = [80000, 120000, 160000]
config.train.warmup_it = -1
config.train.momentum = 0.9
config.train.weight_decay = 1e-4
config.train.max_epoch = 500
config.train.lr_decay_epoch = 8
config.train.display_iter = 20
config.train.summary_iter = 100
config.train.output_path = 'models/nyu_perc_v2.3/'

config.train.resume = False
config.train.snapshot = 'models/nyu_base_v1.0/epoch-0.pth'
config.train.perc_snapshot = 'models/nyu_scene_v1.0/epoch-18.pth'
config.train.pretrained = False
config.train.pretrained_backbone = 'models/resnet-50-caffe.pth'

config.test = edict()
config.test.batch_size = 1
config.test.augment = edict()
config.test.augment.min_size = [228]
config.test.augment.canonical_size = 228
config.test.augment.max_size = [512]


config.gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
config.gpu = ['cuda:%s'%i for i,j in enumerate(config.gpu_id.split(','))]
