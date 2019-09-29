#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Mobilenet models.


from model.backbone.dfnet.dfnet_models import *


class DFNetBackbone(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self, backbone=None, pretrained=None):
        arch = self.configer.get('network.backbone') if backbone is None else backbone
        pretrained = self.configer.get('network.pretrained') if pretrained is None else pretrained

        if arch == 'dfnetv1':
            arch_net = dfnetv1(pretrained=pretrained)

        elif arch == 'dfnetv2':
            arch_net = dfnetv2(pretrained=pretrained)

        else:
            raise Exception('Architecture undefined!')

        return arch_net
