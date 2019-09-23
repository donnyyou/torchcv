#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Mobilenet models.


from model.backbone.mobilenet.mobilenet_models import MobileNetModels


class MobileNetBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.mobile_models = MobileNetModels(self.configer)

    def __call__(self, backbone=None, pretrained=None):
        arch = self.configer.get('network.backbone') if backbone is None else backbone
        pretrained = self.configer.get('network.pretrained') if pretrained is None else pretrained

        if arch == 'mobilenetv2':
            arch_net = self.mobile_models.mobilenetv2(pretrained=pretrained)

        elif arch == 'mobilenetv2_dilated8':
            arch_net = self.mobile_models.mobilenetv2_dilated8(pretrained=pretrained)

        else:
            raise Exception('Architecture undefined!')

        return arch_net
