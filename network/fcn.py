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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#from upsnet.config.config import config
#from upsnet.operators.modules.deform_conv import DeformConv, DeformConvWithOffset
#from upsnet.operators.modules.roialign import RoIAlign
#
class FCNSubNet(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, bn=False, dilation=1):
        super(FCNSubNet, self).__init__()

        self.num_layers = num_layers
        self.conv = nn.ModuleList()
        for i in range(num_layers):
            conv = []
            if i == num_layers - 2:
                conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation))
                in_channels = out_channels
            else:
                conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation))
            if bn:
                conv.append(nn.BatchNorm2d(in_channels))
            conv.append(nn.ReLU(inplace=True))
            self.conv.append(nn.Sequential(*conv))

        #self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()
                #m.weight.data.fill_(0)
                #m.bias.data.fill_(0)
            #elif isinstance(m, DeformConv):
            #    nn.init.kaiming_normal_(m.weight.data)
            #    if m.bias is not None:
            #        m.bias.data.fill_(0)

    def forward(self, x):
        for i in range(self.num_layers):        
            x = self.conv[i](x)
        return x


class Combine(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(Combine, self).__init__()
        if bn:
            self.conv1 = nn.Sequential(nn.Conv2d(2*in_channels, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(2*in_channels, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(2*in_channels, out_channels, 3, padding=1),
                                      nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(2*in_channels, out_channels, 3, padding=1),
                                      nn.ReLU(inplace=True))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def forward(self, x_l, x_h):
        x_h = F.interpolate(x_h, size=x_l.shape[2:], mode='bilinear', align_corners=False)
        x_la = x_l * x_h
        x_lc = torch.cat([x_l, x_la], dim=1)
        x_lc = self.conv1(x_lc)
        x_c = torch.cat([x_h, x_lc], dim=1)
        y = self.conv2(x_c)
        out = y + x_h
        return out


class FCN(nn.Module):

    def __init__(self, feat_channels, num_classes, num_layers, bn=False):
        super(FCN, self).__init__()
        self.fcn_subnet2 = FCNSubNet(feat_channels, feat_channels, num_layers, bn=bn)
        self.fcn_subnet3 = FCNSubNet(feat_channels, feat_channels, num_layers, bn=bn)
        self.fcn_subnet4 = FCNSubNet(feat_channels, feat_channels, num_layers, bn=bn)
        self.fcn_subnet5 = FCNSubNet(feat_channels, feat_channels, num_layers, bn=bn)
        #self.fcn_subnet6 = FCNSubNet(in_channels, feat_channels, num_layers, bn=bn)
        self.comb4 = Combine(feat_channels, feat_channels, bn=bn)
        self.comb3 = Combine(feat_channels, feat_channels, bn=bn)
        self.comb2 = Combine(feat_channels, feat_channels, bn=bn)

        if bn:
            self.score = nn.Sequential(nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False), 
                                       nn.BatchNorm2d(feat_channels), 
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(feat_channels, 1, 3, padding=1),
                                  )
        else:
            self.score = nn.Sequential(nn.Conv2d(feat_channels, feat_channels, 3, padding=1), 
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(feat_channels, 1, 3, padding=1),
                                  )

        #self.initialize()

    def forward(self, fpn_p2, fpn_p3, fpn_p4, fpn_p5, img):
        fpn_p2 = self.fcn_subnet2(fpn_p2)
        fpn_p3 = self.fcn_subnet3(fpn_p3)
        fpn_p4 = self.fcn_subnet4(fpn_p4)
        fpn_p5 = self.fcn_subnet5(fpn_p5)
        #fpn_p6 = self.fcn_subnet6(fpn_p6)

        f4 = self.comb4(fpn_p4, fpn_p5)
        f3 = self.comb3(fpn_p3, f4)
        f2 = self.comb2(fpn_p2, f3)
        score_lr = self.score(f2)
        score = F.interpolate(score_lr, img.shape[2:], mode='bilinear', align_corners=False)
        return score


    def initialize(self):
        nn.init.normal_(self.score.weight.data, 0, 0.01)
        self.score.bias.data.zero_()


