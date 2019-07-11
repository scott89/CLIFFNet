import torch
from torch import nn
from network.resnet import ResNetBackbone
from network.fpn import FPN
from network.fcn import FCN





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.res_backbone = ResNetBackbone([3, 4, 6, 3])
        self.fpn = FPN(bn=True)
        self.fcn = FCN(in_channels=256, num_classes=1, num_layers=3, bn=True)

    def forward(self, x):
        res2, res3, res4, res5 = self.res_backbone(x)
        fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.fpn(res2, res3, res4, res5)
        output = self.fcn(fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6, x)
        return output


