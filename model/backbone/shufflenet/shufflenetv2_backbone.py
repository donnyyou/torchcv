#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch.nn as nn

from model.backbone.shufflenet.shufflenetv2_models import *


class NormalShuffleNetv2Backbone(nn.Module):
    def __init__(self, orig_shufflenet):
        super(NormalShuffleNetv2Backbone, self).__init__()

        self.num_features = orig_shufflenet._stage_out_channels[-1]
        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_shufflenet.conv1
        self.maxpool = orig_shufflenet.maxpool
        self.stage2 = orig_shufflenet.stage2
        self.stage3 = orig_shufflenet.stage3
        self.stage4 = orig_shufflenet.stage4
        self.conv5 = orig_shufflenet.conv5

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x


class ShuffleNetv2Backbone(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self, backbone=None, pretrained=None):
        arch = self.configer.get('network.backbone') if backbone is None else backbone
        pretrained = self.configer.get('network.pretrained') if pretrained is None else pretrained

        if arch == 'shufflenet_v2_x0_5':
            orig_shufflenet = shufflenet_v2_x0_5(pretrained=pretrained)
            arch_net = NormalShuffleNetv2Backbone(orig_shufflenet)

        elif arch == 'shufflenet_v2_x1_0':
            orig_shufflenet = shufflenet_v2_x1_0(pretrained=pretrained)
            arch_net = NormalShuffleNetv2Backbone(orig_shufflenet)

        elif arch == 'shufflenet_v2_x1_5':
            orig_shufflenet = shufflenet_v2_x1_5(pretrained=pretrained)
            arch_net = NormalShuffleNetv2Backbone(orig_shufflenet)

        elif arch == 'shufflenet_v2_x2_0':
            orig_shufflenet = shufflenet_v2_x2_0(pretrained=pretrained)
            arch_net = NormalShuffleNetv2Backbone(orig_shufflenet)

        else:
            raise Exception('Architecture undefined!')

        return arch_net
