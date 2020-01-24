import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneNet(nn.Module):
    def __init__(self, bn=True):
        super(SceneNet, self).__init__()
        self.features = list()
        # stage 1
        self.make_layer(1, 16, bn=bn)
        self.make_layer(16, 16, bn=bn)
        # stage 2
        self.make_layer(16, 32, stride=2, bn=bn)
        self.make_layer(32, 32, bn=bn)
        # stage 3
#        self.make_layer(32, 64, stride=2, bn=bn)
#        self.make_layer(64, 64, bn=bn)
        # 
        self.features = nn.Sequential(*self.features)


    def make_layer(self, in_channels, out_channels, 
                   kernel_size=3, stride=1, padding=1, relu=True, bn=False, negative_slop=1e-2):
        bias = not bn
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.features.append(conv_layer)
        if bn:
            self.features.append(nn.BatchNorm2d(out_channels))
        if relu:
            self.features.append(nn.LeakyReLU(negative_slope=negative_slop))

    def forward(self, x, inter_output_layer=None):
        if inter_output_layer is None:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        else:
            inter_out = []
            for i, l in enumerate(self.features):
                x = l(x)
                if i in inter_output_layer:
                    inter_out.append(x)
            return inter_out


