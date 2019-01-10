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
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=1, dilation=1, relu=True, bn_type=None):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.relu = relu
        self.conv =  nn.Conv2d(in_channels=inplanes,out_channels=outplanes,
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
        pool_scales = (1, 2, 3, 6)
        self.ppm = []
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

        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))

        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)
        aux = self.cbr_deepsup(conv4)
        aux = self.dropout_deepsup(aux)
        aux = self.conv_last_deepsup(aux)

        return x, aux


class EmbedModule(nn.Module):
    def __init__(self, inchannels, bn_type):
        super(EmbedModule, self).__init__()
        inter_channels = inchannels // 4
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, inter_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=7, padding=3, stride=1, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inchannels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(inchannels),
            nn.ReLU(inplace=True),
        )
        # downsample & conv to get embedding.

    def forward(self, x):
        # return incremental for features & embedding for loss function.
        x = self.conv(x)
        return x


class EmbedNet(nn.Sequential):
    def __init__(self, configer):
        super(EmbedNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        num_features = self.backbone.get_num_features()

        if 'caffe' in self.configer.get('network', 'backbone'):
            self.low_features = nn.Sequential(
                self.backbone.conv1, self.backbone.bn1, self.backbone.relu1,
                self.backbone.conv2, self.backbone.bn2, self.backbone.relu2,
                self.backbone.conv3, self.backbone.bn3, self.backbone.relu3,
                self.backbone.maxpool,
                self.backbone.layer1,
            )
        else:
            self.low_features = nn.Sequential(
                self.backbone.conv1, self.backbone.bn1, self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,
            )

        self.high_features1 = nn.Sequential(self.backbone.layer2, self.backbone.layer3)
        self.high_features2 = nn.Sequential(self.backbone.layer4)
        self.embed_conv = EmbedModule(1024, bn_type=self.configer.get('network', 'bn_type'))
        self.decoder = PPMBilinearDeepsup(num_class=self.num_classes, fc_dim=num_features,
                                          bn_type=self.configer.get('network', 'bn_type'))

    def forward(self, x):
        low = self.low_features(x)
        aux = self.high_features1(low)
        incr = self.embed_conv(aux)
        x = self.high_features2(aux+incr)
        x, aux = self.decoder([x, aux])
        x = F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=False)

        return x, aux, incr


if __name__ == '__main__':
    i = torch.Tensor(1,3,512,512).cuda()
    model = EmbedNet(num_classes=19).cuda()
    model.eval()
    o, _ = model(i)
    #print(o.size())
    #final_out = F.upsample(o,scale_factor=8)
    #print(final_out.size())
    print(o.size())
    print(_.size())
