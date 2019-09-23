#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# VGG models.


from model.backbone.vgg.vgg_models import VGGModels


class VGGBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.vgg_models = VGGModels(self.configer)

    def __call__(self, backbone=None, pretrained=None):
        arch = self.configer.get('network.backbone') if backbone is None else backbone
        pretrained = self.configer.get('network.pretrained') if pretrained is None else pretrained
        if 'bn' in arch:
            arch_net = self.vgg_models.vgg_bn(backbone=arch, pretrained=pretrained)

        else:
            arch_net = self.vgg_models.vgg(backbone=arch, pretrained=pretrained)

        return arch_net
