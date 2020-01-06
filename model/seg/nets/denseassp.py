#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# SynBN_version of DenseAspp


import torch
from torch import nn
import torch.nn.functional as F

from lib.model.module_helper import ModuleHelper
from model.seg.loss.loss import BASE_LOSS_DICT


MODEL_CONFIG = {
    'dropout0': 0.1,
    'dropout1': 0.1,
    'd_feature0': 256,
    'd_feature1': 128
}


class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """
    def __init__(self, configer):
        super(DenseASPP, self).__init__()
        self.configer = configer

        dropout0 = MODEL_CONFIG['dropout0']
        dropout1 = MODEL_CONFIG['dropout1']
        d_feature0 = MODEL_CONFIG['d_feature0']
        d_feature1 = MODEL_CONFIG['d_feature1']

        self.backbone = ModuleHelper.get_backbone(
            backbone=self.configer.get('network.backbone'),
            pretrained=self.configer.get('network.pretrained')
        )

        num_features = self.backbone.get_num_features()

        self.trans = _Transition(num_input_features=self.num_features,
                                 num_output_features=self.num_features // 2,
                                 norm_type=self.configer.get('network', 'norm_type'))

        self.num_features = self.num_features // 2

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0,
                                      norm_type=self.configer.get('network', 'norm_type'))

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0,
                                      norm_type=self.configer.get('network', 'norm_type'))

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0,
                                       norm_type=self.configer.get('network', 'norm_type'))

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0,
                                       norm_type=self.configer.get('network', 'norm_type'))

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0,
                                       norm_type=self.configer.get('network', 'norm_type'))

        num_features = num_features + 5 * d_feature1

        self.classification = nn.Sequential(
            nn.Dropout2d(p=dropout1),
            nn.Conv2d(num_features, self.configer.get('data', 'num_classes'), kernel_size=1, padding=0)
        )
        self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))

    def forward(self, data_dict):
        x = self.backbone(data_dict['img'])
        feature = self.trans(x)

        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        x = self.classification(feature)

        x = F.interpolate(x, size=(data_dict['img'].size(2), data_dict['img'].size(3)),
                          mode="bilinear", align_corners=True)
        out_dict = dict(out=x)
        if self.configer.get('phase') == 'test':
            return out_dict

        loss_dict = dict()
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


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, norm_type):
        super(_DenseAsppBlock, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm1', ModuleHelper.BatchNorm2d(norm_type=norm_type)(num_features=num1)),
        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),
        self.add_module('norm2', ModuleHelper.BatchNorm2d(norm_type=norm_type)(num_features=input_num)),
        self.add_module('relu2', nn.ReLU(inplace=False)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_type):
        super(_Transition, self).__init__()
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('norm', ModuleHelper.BatchNorm2d(norm_type=norm_type)(num_features=num_output_features)),
        self.add_module('relu', nn.ReLU(inplace=False))


if __name__ == "__main__":
    model = DenseASPP(12)
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    out = model(image)
    print(out.size())
