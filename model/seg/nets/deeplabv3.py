#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# deeplabv3 res101 (synchronized BN version)


import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.module_helper import ModuleHelper
from model.seg.loss.loss import BASE_LOSS_DICT


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=512, out_features=512, dilations=(12, 24, 36), norm_type=None):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   ModuleHelper.BNReLU(inner_features, norm_type=norm_type))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(inner_features, norm_type=norm_type))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            ModuleHelper.BNReLU(inner_features, norm_type=norm_type))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            ModuleHelper.BNReLU(inner_features, norm_type=norm_type))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            ModuleHelper.BNReLU(inner_features, norm_type=norm_type))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features, norm_type=norm_type),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=False)

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
        base = ModuleHelper.get_backbone(
            backbone=self.configer.get('network.backbone'),
            pretrained=self.configer.get('network.pretrained')
        )
        self.stage1 = nn.Sequential(
            base.conv1, base.bn1, base.relu1, base.conv2, base.bn2, base.relu2, base.conv3, base.bn3,
            base.relu3, base.maxpool, base.layer1, base.layer2, base.layer3
        )
        self.stage2 = base.layer4
        num_features = 512 if 'resnet18' in self.configer.get('network.backbone') else 2048
        self.head = nn.Sequential(ASPPModule(num_features, norm_type=self.configer.get('network', 'norm_type')),
                                  nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, norm_type=self.configer.get('network', 'norm_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))

    def forward(self, data_dict):
        x = self.stage1(data_dict['img'])
        x_dsn = self.dsn(x)
        x = self.stage2(x)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(data_dict['img'].size(2), data_dict['img'].size(3)),
                              mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(data_dict['img'].size(2), data_dict['img'].size(3)),
                          mode="bilinear", align_corners=False)
        out_dict = dict(dsn_out=x_dsn, out=x)
        if self.configer.get('phase') == 'test':
            return out_dict

        loss_dict = dict()
        if 'dsn_ce_loss' in self.valid_loss_dict:
            loss_dict['dsn_ce_loss'] = dict(
                params=[x_dsn, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['dsn_ce_loss']])
            )

        if 'ce_loss' in self.valid_loss_dict:
            loss_dict['ce_loss'] = dict(
                params=[x, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_loss']])
            )

        if 'ohem_ce_loss' in self.valid_loss_dict:
            loss_dict['ohem_ce_loss'] = dict(
                params=[x, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ohem_ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ohem_ce_loss']])
            )
        return out_dict, loss_dict


if __name__ == '__main__':
    model = DeepLabV3(20, multi_grid=[1, 2, 1])
    model.freeze_bn()
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True)
    print(type(model.resnet_features))
    print (model(image).size())
