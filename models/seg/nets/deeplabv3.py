#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# deeplabv3 res101 (synchronized BN version)


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import ModuleHelper


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=512, out_features=512, dilations=(12, 24, 36), bn_type=None):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


class DeepLabV3(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(DeepLabV3, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        self.head = nn.Sequential(ASPPModule(2048, bn_type=self.configer.get('network', 'bn_type')),
                                  nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn(x[-2])
        x = self.head(x[-1])
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


if __name__ == '__main__':
    model = DeepLabv3(20, multi_grid=[1, 2, 1])
    model.freeze_bn()
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True)
    print(type(model.resnet_features))
    print (model(image).size())
