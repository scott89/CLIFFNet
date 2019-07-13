# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from core.config import config



def get_params(model, prefixs, suffixes, exclude=None):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    for name, module in model.named_modules():
        for prefix in prefixs:
            if name == prefix:
                for n, p in module.named_parameters():
                    n = '.'.join([name, n])
                    if type(exclude) == list and n in exclude:
                        continue
                    if type(exclude) == str and exclude in n:
                        continue

                    for suffix in suffixes:
                        if (n.split('.')[-1].startswith(suffix) or n.endswith(suffix)) and p.requires_grad:
                            yield p
                break

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fix_bn=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if fix_bn:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()
            for i in self.bn1.parameters():
                i.requires_grad = False
            for i in self.bn2.parameters():
                i.requires_grad = False
            for i in self.bn3.parameters():
                i.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class conv1(nn.Module):
    def __init__(self, requires_grad=False):
        super(conv1, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if not requires_grad:
            self.eval()
            for i in self.parameters():
                i.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class res_block(nn.Module):
    def __init__(self, planes, blocks, block=Bottleneck, stride=1, dilation=1, fix_bn=True, with_dpyramid=False):
        super(res_block, self).__init__()
        downsample = None
        self.inplanes = planes * 2 if planes != 64 else planes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
                
            if fix_bn:
                downsample[1].eval()
                for i in downsample[1].parameters():
                    i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, fix_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes, dilation=dilation, fix_bn=fix_bn))
        layers.append(block(self.inplanes, planes, dilation=dilation, fix_bn=fix_bn))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

        
class ResNetBackbone(nn.Module):

    def __init__(self, blocks):
        super(ResNetBackbone, self).__init__()

        self.fix_bn = config.network.backbone_fix_bn
        self.with_dilation = config.network.backbone_with_dilation
        self.freeze_at = config.network.backbone_freeze_at


        self.conv1 = conv1(requires_grad=True)
        self.res2 = res_block(64, blocks[0], fix_bn=self.fix_bn)
        self.res3 = res_block(128, blocks[1], block= Bottleneck,
                              stride=2, fix_bn=self.fix_bn)
        self.res4 = res_block(256, blocks[2], block= Bottleneck,
                              stride=2, fix_bn=self.fix_bn)
        if self.with_dilation:
            res5_stride, res5_dilation = 1, 2
        else:
            res5_stride, res5_dilation = 2, 1
        self.res5 = res_block(512, blocks[3], block= Bottleneck,
                              stride=res5_stride, dilation=res5_dilation, fix_bn=self.fix_bn)
        if self.freeze_at > 0:
            for p in self.conv1.parameters():
                p.requires_grad = False
            self.conv1.eval()
            for i in range(2, self.freeze_at + 1):
                for p in eval('self.res{}'.format(i)).parameters():
                    p.requires_grad = False
                eval('self.res{}'.format(i)).eval()

    def modify_state_dict_keys(self, state_dict):
        new_state_dict = dict()
        for key, value in state_dict.items():
            if key.startswith('conv1') or key.startswith('bn1'):
                key = 'conv1.' + key
                new_state_dict.update({key: value})
            else:
                key = key.replace('layer1', 'res2.layers') \
                        .replace('layer2', 'res3.layers') \
                        .replace('layer3', 'res4.layers') \
                        .replace('layer4', 'res5.layers')
                new_state_dict.update({key: value})
        return new_state_dict

    def forward(self, x):

        conv1 = self.conv1(x).detach() if self.freeze_at == 1 else self.conv1(x)
        res2 = self.res2(conv1).detach() if self.freeze_at == 2 else self.res2(conv1)

        res3 = self.res3(res2).detach() if self.freeze_at == 3 else self.res3(res2)
        res4 = self.res4(res3).detach() if self.freeze_at == 4 else self.res4(res3)
        res5 = self.res5(res4).detach() if self.freeze_at == 5 else self.res5(res4)

        return res2, res3, res4, res5
    def freeze_bn_at_initial(self):
        if self.freeze_at > 0:
            self.conv1.eval()
            for i in range(2, self.freeze_at + 1):
                eval('self.res{}'.format(i)).eval()

