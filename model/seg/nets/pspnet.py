#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Pytorch implementation of PSP net Synchronized Batch Normalization
# this is pytorch implementation of PSP resnet101 (syn-bn) version


import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.module_helper import ModuleHelper
from model.seg.loss.loss import BASE_LOSS_DICT


class _ConvBatchNormReluBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=1, dilation=1, norm_type=None):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=inplanes,out_channels=outplanes,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation = dilation, bias=False)
        self.bn_relu = ModuleHelper.BNReLU(outplanes, norm_type=norm_type)

    def forward(self, x):
        x = self.bn_relu(self.conv(x))
        return x


# PSP decoder Part
# pyramid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    def __init__(self, fc_dim=4096, norm_type=None):
        super(PPMBilinearDeepsup, self).__init__()
        self.norm_type = norm_type
        pool_scales = (1, 2, 3, 6)
        self.ppm = []
        # assert norm_type == 'syncbn' or not self.training
        # Torch BN can't handle feature map size with 1x1.
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                ModuleHelper.BNReLU(512, norm_type=norm_type)
            ))

        self.ppm = nn.ModuleList(self.ppm)

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(pool_scale(x), (input_size[2], input_size[3]),
                                         mode='bilinear', align_corners=False))

        ppm_out = torch.cat(ppm_out, 1)

        return ppm_out


class PSPNet(nn.Sequential):
    def __init__(self, configer):
        super(PSPNet, self).__init__()
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
        self.dsn = nn.Sequential(
            _ConvBatchNormReluBlock(num_features // 2, num_features // 4, 3, 1,
                                    norm_type=self.configer.get('network', 'norm_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(num_features // 4, self.num_classes, 1, 1, 0)
        )
        self.ppm = PPMBilinearDeepsup(fc_dim=num_features, norm_type=self.configer.get('network', 'norm_type'))

        self.cls = nn.Sequential(
            nn.Conv2d(num_features + 4 * 512, 512, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(512, norm_type=self.configer.get('network', 'norm_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1)
        )
        self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))

    def forward(self, data_dict):
        x = self.stage1(data_dict['img'])
        aux_x = self.dsn(x)
        x = self.stage2(x)
        x = self.ppm(x)
        x = self.cls(x)
        x_dsn = F.interpolate(aux_x, size=(data_dict['img'].size(2), data_dict['img'].size(3)),
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
    i = torch.Tensor(1,3,512,512).cuda()
    model = PSPNet(num_classes=19).cuda()
    model.eval()
    o, _ = model(i)
    #print(o.size())
    #final_out = F.upsample(o,scale_factor=8)
    #print(final_out.size())
    print(o.size())
    print(_.size())
