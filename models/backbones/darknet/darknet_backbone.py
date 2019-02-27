#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch.nn as nn

from models.backbones.darknet.darknet_models import DarkNetModels


class NormalDarknetBackbone(nn.Module):
    def __init__(self, orig_darknet):
        super(NormalDarknetBackbone, self).__init__()

        self.num_features = [64, 128, 256, 512, 1024]
        # take pretrained darknet, except AvgPool and FC
        self.conv1 = orig_darknet.conv1
        self.bn1 = orig_darknet.bn1
        self.relu1 = orig_darknet.relu1
        self.layer1 = orig_darknet.layer1
        self.layer2 = orig_darknet.layer2
        self.layer3 = orig_darknet.layer3
        self.layer4 = orig_darknet.layer4
        self.layer5 = orig_darknet.layer5

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)
        x = self.layer5(x)
        tuple_features.append(x)

        return tuple_features


class DilatedDarknetBackbone(nn.Module):
    def __init__(self, orig_darknet, dilate_scale=8):
        super(DilatedDarknetBackbone, self).__init__()

        self.num_features = [64, 128, 256, 512, 1024]
        from functools import partial

        if dilate_scale == 8:
            orig_darknet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_darknet.layer5.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_darknet.layer5.apply(
                partial(self._nostride_dilate, dilate=2))

        # Take pretrained darknet, except AvgPool and FC
        self.conv1 = orig_darknet.conv1
        self.bn1 = orig_darknet.bn1
        self.relu1 = orig_darknet.relu1
        self.layer1 = orig_darknet.layer1
        self.layer2 = orig_darknet.layer2
        self.layer3 = orig_darknet.layer3
        self.layer4 = orig_darknet.layer4
        self.layer5 = orig_darknet.layer5

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)
        x = self.layer5(x)
        tuple_features.append(x)

        return tuple_features


class DarkNetBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.darknet_models = DarkNetModels(self.configer)

    def __call__(self):
        arch = self.configer.get('network', 'backbone')
        if arch == 'darknet21':
            orig_darknet = self.darknet_models.darknet21()
            arch_net = NormalDarknetBackbone(orig_darknet)

        elif arch == 'darknet21_dilated8':
            orig_darknet = self.darknet_models.darknet21()
            arch_net = DilatedDarknetBackbone(orig_darknet, dilate_scale=8)

        elif arch == 'darknet21_dilated16':
            orig_darknet = self.darknet_models.darknet21()
            arch_net = DilatedDarknetBackbone(orig_darknet, dilate_scale=16)

        elif arch == 'darknet53':
            orig_darknet = self.darknet_models.darknet53()
            arch_net = NormalDarknetBackbone(orig_darknet)

        elif arch == 'darknet53_dilated8':
            orig_darknet = self.darknet_models.darknet53()
            arch_net = DilatedDarknetBackbone(orig_darknet, dilate_scale=8)

        elif arch == 'darknet53_dilated16':
            orig_darknet = self.darknet_models.darknet53()
            arch_net = DilatedDarknetBackbone(orig_darknet, dilate_scale=16)

        else:
            raise Exception('Architecture undefined!')

        return arch_net
