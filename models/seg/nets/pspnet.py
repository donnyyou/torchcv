#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Pytorch implementation of PSP net Synchronized Batch Normalization
# this is pytorch implementation of PSP resnet101 (syn-bn) version


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import ModuleHelper


class _ConvBatchNormReluBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=1, dilation=1, bn_type=None):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=inplanes,out_channels=outplanes,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation = dilation, bias=False)
        self.bn_relu = ModuleHelper.BNReLU(outplanes, bn_type=bn_type)

    def forward(self, x):
        x = self.bn_relu(self.conv(x))
        return x


# PSP decoder Part
# pyramid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    def __init__(self, fc_dim=4096, bn_type=None):
        super(PPMBilinearDeepsup, self).__init__()
        self.bn_type = bn_type
        pool_scales = (1, 2, 3, 6)
        self.ppm = []
        # assert bn_type == 'syncbn' or not self.training
        # Torch BN can't handle feature map size with 1x1.
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                ModuleHelper.BNReLU(512, bn_type=bn_type)
            ))

        self.ppm = nn.ModuleList(self.ppm)

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        assert not (self.bn_type == 'torchbn' and self.training and x.size(0) == 1)
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(pool_scale(x), (input_size[2], input_size[3]),
                                         mode='bilinear', align_corners=True))

        ppm_out = torch.cat(ppm_out, 1)

        return ppm_out


class PSPNet(nn.Sequential):
    def __init__(self, configer):
        super(PSPNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        num_features = self.backbone.get_num_features()
        self.dsn = nn.Sequential(
            _ConvBatchNormReluBlock(num_features // 2, num_features // 4, 3, 1,
                                    bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(num_features // 4, self.num_classes, 1, 1, 0)
        )
        self.ppm = PPMBilinearDeepsup(fc_dim=num_features, bn_type=self.configer.get('network', 'bn_type'))

        self.cls = nn.Sequential(
            nn.Conv2d(num_features + 4 * 512, 512, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.ppm(x[-1])
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return aux_x, x


if __name__ == '__main__':
    i = torch.Tensor(1,3,512,512).cuda()
    model = PSPNet(num_classes=19).cuda()
    model.eval()
    o, _ = model(i)
    #print(o.size())
    #final_out = F.upsample(o,scale_factor=8)
    #print(final_out.size())
    print(o.size())
    print(_.size())
