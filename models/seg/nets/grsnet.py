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


class ContextResidualModule(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, stride=1):
        super(ContextResidualModule, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, inter_channels, (3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2_1 = nn.Conv2d(inter_channels, out_channels, (7, 1), stride=1, padding=(3, 0), bias=False)

        self.conv2_2 = nn.Conv2d(out_channels, out_channels, (1, 7), stride=1, padding=(0, 3), bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        output = self.conv1(input)
        output = F.relu(output)
        output = self.bn1(output)

        output = self.conv2_1(output)
        output = F.relu(output)
        output = self.conv2_2(output)
        output = self.bn2(output)

        input = F.max_pool2d(input, kernel_size=self.stride)
        return F.relu(output + input)  # +input = identity (residual connection)


class GatedResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedResidualModule, self).__init__()



class _ConvBatchNormReluBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=1, dilation=1, relu=True, bn_type=None):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_channels=inplanes,out_channels=outplanes,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation = dilation, bias=False)
        self.bn = ModuleHelper.BatchNorm2d(bn_type=bn_type)(num_features=outplanes)
        self.relu_f = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.relu:
            x = self.relu_f(x)
        return x


# PSP decoder Part
# pyramid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, bn_type=None):
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
                ModuleHelper.BatchNorm2d(bn_type=bn_type)(512),
                nn.ReLU(inplace=True)
            ))

        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = _ConvBatchNormReluBlock(fc_dim // 2, fc_dim // 4, 3, 1, bn_type=bn_type)
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out):
        conv5, conv4 = conv_out
        input_size = conv5.size()
        ppm_out = [conv5]
        assert self.bn_type == 'syncbn' or not self.training or conv5.size(0) > 1
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(pool_scale(conv5),
                                         (input_size[2], input_size[3]),
                                         mode='bilinear', align_corners=False))

        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)
        aux = self.cbr_deepsup(conv4)
        aux = self.dropout_deepsup(aux)
        aux = self.conv_last_deepsup(aux)

        return x, aux


class PSPNet(nn.Sequential):
    def __init__(self, configer):
        super(PSPNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        num_features = self.backbone.get_num_features()

        self.low_features = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
        )

        self.high_features1 = nn.Sequential(self.backbone.layer2, self.backbone.layer3)
        self.high_features2 = nn.Sequential(self.backbone.layer4)
        self.decoder = PPMBilinearDeepsup(num_class=self.num_classes, fc_dim=num_features,
                                          bn_type=self.configer.get('network', 'bn_type'))

    def forward(self, x_):
        low = self.low_features(x_)
        aux = self.high_features1(low)
        x = self.high_features2(aux)
        x, aux = self.decoder([x, aux])
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=False)

        return x, aux


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