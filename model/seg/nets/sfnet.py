#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@pku.edu.cn), Xiangtai Li(lxt@pku.edu.cn)


import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.model.module_helper import ModuleHelper
from model.seg.loss.loss import BASE_LOSS_DICT


def conv3x3_bn_relu(in_planes, out_planes, stride=1, norm_type="batchnorm"):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            ModuleHelper.BNReLU(out_planes, norm_type=norm_type),
    )


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_type="batchnorm"):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_type) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features, norm_type=norm_type),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm_type):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = ModuleHelper.BatchNorm2d(norm_type=norm_type)(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class AlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature= x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class AlignHead(nn.Module):
    def __init__(self, inplanes, norm_type="batchnorm", fpn_dim=256):
        super(AlignHead, self).__init__()
        self.ppm = PSPModule(inplanes, norm_type=norm_type, out_features=fpn_dim)
        fpn_inplanes = [inplanes // 8, inplanes// 4, inplanes // 2, inplanes]
        self.fpn_in = nn.ModuleList()
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    ModuleHelper.BNReLU(fpn_dim, norm_type=norm_type),
                )
            )

        self.fpn_out = nn.ModuleList()
        self.fpn_out_align = nn.ModuleList()
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
            self.fpn_out_align.append(
                AlignModule(inplane=fpn_dim, outplane=fpn_dim//2)
            )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            out.append(f)

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))

        fusion_out = torch.cat(fusion_list, 1)
        return fusion_out, out


class ResSFNet(nn.Module):
    """
        Resnet-like Graph SegNet
    """
    def __init__(self, configer):
        super(ResSFNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        base = ModuleHelper.get_backbone(
            backbone=self.configer.get('network.backbone'),
            pretrained=self.configer.get('network.pretrained')
        )
        self.stage1 = nn.Sequential(
            base.conv1, base.bn1, base.relu1, base.conv2, base.bn2, base.relu2, base.conv3, base.bn3,
            base.relu3, base.maxpool, base.layer1
        )
        self.stage2 = base.layer2
        self.stage3 = base.layer3
        self.stage4 = base.layer4
        num_features = 512 if 'resnet18' in self.configer.get('network.backbone') else 2048
        fpn_dim = max(num_features // 8, 128)
        self.head = AlignHead(num_features, fpn_dim=fpn_dim)
        self.dsn = nn.Sequential(
            nn.Conv2d(num_features // 2, max(num_features // 4, 256), kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(max(num_features // 4, 256), norm_type="batchnorm"),
            nn.Dropout2d(0.1),
            nn.Conv2d(max(num_features // 4, 256), self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(4 * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.num_classes, kernel_size=1)
        )
        self.fpn_dsn = nn.ModuleList()
        for i in range(len([2, 4, 8])):
            self.fpn_dsn.append(
                nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                    ModuleHelper.BNReLU(fpn_dim, norm_type="batchnorm"),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(fpn_dim, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
            )

        self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))

    def forward(self, data_dict):
        target_size = (data_dict['img'].size(2), data_dict['img'].size(3))
        x1 = self.stage1(data_dict['img'])
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x_ = [x1, x2, x3, x4]
        x, fpn_dsn = self.head(x_)
        x = self.conv_last(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        out_dict = dict(out=x)
        if self.configer.get('phase') == 'test':
            return out_dict

        x_dsn = self.dsn(x_[-2])
        x_dsn = F.interpolate(x_dsn, size=target_size, mode="bilinear", align_corners=False)
        loss_dict = dict()
        for i in range(len(self.fpn_dsn)):
            fpn_out = self.fpn_dsn[i](fpn_dsn[i])
            fpn_out = F.interpolate(fpn_out, size=target_size, mode="bilinear", align_corners=False)

            if 'fpn_ce_loss{}'.format(i) in self.valid_loss_dict:
                loss_dict['fpn_ce_loss{}'.format(i)] = dict(
                    params=[fpn_out, data_dict['labelmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['fpn_ce_loss{}'.format(i)]])
                )

            if 'fpn_ohem_ce_loss{}'.format(i) in self.valid_loss_dict:
                loss_dict['fpn_ohem_ce_loss{}'.format(i)] = dict(
                    params=[fpn_out, data_dict['labelmap']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['ohem_ce_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['fpn_ohem_ce_loss{}'.format(i)]])
                )

        if 'dsn_ce_loss' in self.valid_loss_dict:
            loss_dict['dsn_ce_loss'] = dict(
                params=[x_dsn, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['dsn_ce_loss']])
            )

        if 'dsn_ohem_ce_loss' in self.valid_loss_dict:
            loss_dict['dsn_ohem_ce_loss'] = dict(
                params=[x_dsn, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ohem_ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['dsn_ohem_ce_loss']])
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
