#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# deeplabv3 res101 (synchronized BN version)


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import ModuleHelper


class _ConvBatchNormReluBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, dilation, relu=True, bn_type=None):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.relu = relu
        self.conv =  nn.Conv2d(in_channels=inplanes,out_channels=outplanes,
                            kernel_size=kernel_size, stride=stride, padding = padding,
                            dilation = dilation, bias=False)
        self.bn = ModuleHelper.BatchNorm2d(bn_type=bn_type)(num_features=outplanes)
        self.relu_f = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.relu:
            x = self.relu_f(x)

        return x


class _Bottleneck(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, stride, dilation, downsample, bn_type):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBatchNormReluBlock(inplanes, midplanes, 1, stride, 0, 1, bn_type=bn_type)
        self.conv3x3 = _ConvBatchNormReluBlock(midplanes, midplanes, 3, 1, dilation, dilation, bn_type=bn_type)
        self.increase = _ConvBatchNormReluBlock(midplanes, outplanes, 1, 1, 0, 1, relu=False, bn_type=bn_type)
        self.downsample = downsample
        if self.downsample:
            self.proj = _ConvBatchNormReluBlock(inplanes, outplanes, 1, stride, 0, 1, relu=False, bn_type=bn_type)

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class _ResidualBlockMulGrid(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, stride, dilation, mulgrid=[1,2,1], bn_type=None):
        super(_ResidualBlockMulGrid, self).__init__()
        self.block1 = _Bottleneck(inplanes, midplanes, outplanes, stride, dilation * mulgrid[0], True, bn_type)
        self.block2 = _Bottleneck(outplanes, midplanes, outplanes, 1, dilation * mulgrid[1], False, bn_type)
        self.block3 = _Bottleneck(outplanes, midplanes, outplanes, 1, dilation * mulgrid[2], False, bn_type)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module with image pool (Deeplabv3)"""

    def __init__(self, in_channels, out_channels, pyramids, bn_type):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module(
            'c0',
            _ConvBatchNormReluBlock(in_channels, out_channels, 1, 1, 0, 1, bn_type=bn_type),
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                'c{}'.format(i + 1),
                _ConvBatchNormReluBlock(in_channels, out_channels, 3, 1, padding, dilation, bn_type=bn_type),
            )
        self.imagepool = nn.Sequential(
                 nn.AdaptiveAvgPool2d(1),
                _ConvBatchNormReluBlock(in_channels, out_channels, 1, 1, 0, 1, bn_type=bn_type)
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h


class DeepLabV3(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3, self).__init__()
        self.configer = configer
        self.backbone = BackboneSelector(configer).get_backbone()
        
        self.backbone = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu1,
            self.backbone.conv2, self.backbone.bn2, self.backbone.relu2,
            self.backbone.conv3, self.backbone.bn3, self.backbone.relu3, self.backbone.maxpool,
            self.backbone.layer1, self.backbone.layer2, self.backbone.layer3
        )
        self.MG_features = _ResidualBlockMulGrid(inplanes=1024, midplanes=512,
                                                 outplanes=2048, stride=1,
                                                 dilation=2, mulgrid=self.configer.get('network', 'multi_grid'),
                                                 bn_type=self.configer.get('network', 'bn_type'))
        pyramids = [6, 12, 18]
        self.aspp = _ASPPModule(2048, 256, pyramids, bn_type=self.configer.get('network', 'bn_type'))

        self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1),  # 256 * 5 = 1280
                                 ModuleHelper.BatchNorm2d(bn_type=self.configer.get('network', 'bn_type'))(256))
        self.fc2 = nn.Conv2d(256, self.configer.get('data', 'num_classes'), kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.MG_features(x)
        x = self.aspp(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.interpolate(x, scale_factor=(16, 16), mode="bilinear", align_corners=False)
        return x


if __name__ == '__main__':
    model = DeepLabV3(20, multi_grid=[1, 2, 1])
    model.freeze_bn()
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True)
    print(type(model.resnet_features))
    print (model(image).size())
