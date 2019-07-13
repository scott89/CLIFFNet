import torch
from torch import nn
from network.resnet import ResNetBackbone
from network.fpn import FPN
from network.fcn import FCN
from core.config import config





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.res_backbone = ResNetBackbone([3, 4, 6, 3])
        self.fpn = FPN(bn=False)
        self.fcn = FCN(in_channels=256, num_classes=1, num_layers=3, bn=False)

    def forward(self, x):
        res2, res3, res4, res5 = self.res_backbone(x)
        fpn_p2, fpn_p3, fpn_p4, fpn_p5 = self.fpn(res2, res3, res4, res5)
        output = self.fcn(fpn_p2, fpn_p3, fpn_p4, fpn_p5, x)
        return output

    def set_stage(self, stage):
        if stage == 'eval':
            self.eval()
        elif stage == 'train':
            if config.network.backbone_fix_bn:
                self.eval()
            else:
                self.train()
                self.res_backbone.freeze_bn_at_initial()


