import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, feature_dim=256, bn=False):
        super(FPN, self).__init__()
        self.feature_dim = feature_dim
        self.upsample_method = 'bilinear'
        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=self.upsample_method, align_corners=False if self.upsample_method == 'bilinear' else None)
        self.fpn_upsample = interpolate
        #self.fpn_p6 = nn.MaxPool2d(kernel_size=1, stride=2)
        if bn:
            self.fpn_p5_1x1 = nn.Sequential(nn.Conv2d(2048, feature_dim, 1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True))
            self.fpn_p4_1x1 = nn.Sequential(nn.Conv2d(1024, feature_dim, 1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True))
            self.fpn_p3_1x1 = nn.Sequential(nn.Conv2d(512, feature_dim, 1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True))
            self.fpn_p2_1x1 = nn.Sequential(nn.Conv2d(256, feature_dim, 1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True))
            self.fpn_p5 = nn.Sequential(nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True))
            self.fpn_p4 = nn.Sequential(nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True))
            self.fpn_p3 = nn.Sequential(nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True))
            self.fpn_p2 = nn.Sequential(nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True))

        else:
            self.fpn_p5_1x1 = nn.Conv2d(2048, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(1024, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(512, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(256, feature_dim, 1)
            self.fpn_p5 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p4 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p3 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p2 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)

       # self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res2, res3, res4, res5):
        fpn_p5_1x1 = self.fpn_p5_1x1(res5)
        fpn_p4_1x1 = self.fpn_p4_1x1(res4)
        fpn_p3_1x1 = self.fpn_p3_1x1(res3)
        fpn_p2_1x1 = self.fpn_p2_1x1(res2)
        fpn_p5_upsample = F.interpolate(fpn_p5_1x1, size = fpn_p4_1x1.shape[2:], mode=self.upsample_method)
        fpn_p4_plus = fpn_p5_upsample + fpn_p4_1x1
        fpn_p4_upsample = F.interpolate(fpn_p4_plus, size = fpn_p3_1x1.shape[2:], mode=self.upsample_method)
        fpn_p3_plus = fpn_p4_upsample + fpn_p3_1x1
        fpn_p3_upsample = F.interpolate(fpn_p3_plus, size = fpn_p2_1x1.shape[2:], mode=self.upsample_method)
        fpn_p2_plus = fpn_p3_upsample + fpn_p2_1x1

        fpn_p5 = self.fpn_p5(fpn_p5_1x1)
        fpn_p4 = self.fpn_p4(fpn_p4_plus)
        fpn_p3 = self.fpn_p3(fpn_p3_plus)
        fpn_p2 = self.fpn_p2(fpn_p2_plus)
        #fpn_p6 = self.fpn_p6(fpn_p5)
        return fpn_p2, fpn_p3, fpn_p4, fpn_p5#, fpn_p6
