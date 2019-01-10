#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# SynBN_version of DenseAspp


import torch
from torch import nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector
from models.tools.module_helper import ModuleHelper


class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """
    def __init__(self, configer):
        super(DenseASPP, self).__init__()
        self.configer = configer
        dropout0 = 0.1
        dropout1 = 0.1

        self.backbone = BackboneSelector(configer).get_backbone()

        num_features = self.backbone.get_num_features()

        self.trans = _Transition(num_input_features=self.num_features,
                                 num_output_features=self.num_features // 2,
                                 bn_type=self.configer.get('network', 'bn_type'))

        self.num_features = self.num_features // 2

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=256, num2=64,
                                      dilation_rate=3, drop_out=dropout0,
                                      bn_type=self.configer.get('network', 'bn_type'))

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + 64 * 1, num1=256, num2=64,
                                      dilation_rate=6, drop_out=dropout0,
                                      bn_type=self.configer.get('network', 'bn_type'))

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + 64 * 2, num1=256, num2=64,
                                       dilation_rate=12, drop_out=dropout0,
                                       bn_type=self.configer.get('network', 'bn_type'))

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + 64 * 3, num1=256, num2=64,
                                       dilation_rate=18, drop_out=dropout0,
                                       bn_type=self.configer.get('network', 'bn_type'))

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + 64 * 4, num1=256, num2=64,
                                       dilation_rate=24, drop_out=dropout0,
                                       bn_type=self.configer.get('network', 'bn_type'))

        num_features = num_features + 5 * 64

        self.classification = nn.Sequential(
            nn.Dropout2d(p=dropout1),
            nn.Conv2d(in_channels=num_features,
                      out_channels=self.configer.get('network', 'out_channels'), kernel_size=1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

            elif isinstance(m, ModuleHelper.BatchNorm2d(bn_type=self.configer.get('network', 'bn_type'))):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feature = self.backbone(x)

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

        cls = self.classification(feature)

        out = F.interpolate(cls, scale_factor=8, mode='bilinear', align_corners=False)

        return out


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_type):
        super(_DenseAsppBlock, self).__init__()
        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', ModuleHelper.BatchNorm2d(bn_type=bn_type)(num_features=num1)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),
        self.add_module('norm2', ModuleHelper.BatchNorm2d(bn_type=bn_type)(num_features=input_num)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)
        return feature


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, bn_type):
        super(_Transition, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('norm', ModuleHelper.BatchNorm2d(bn_type=bn_type)(num_features=num_output_features)),


if __name__ == "__main__":
    model = DenseASPP(12)
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    out = model(image)
    print(out.size())
