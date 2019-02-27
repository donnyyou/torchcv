#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# VGG models.


from models.backbones.vgg.vgg_models import VGGModels


class VGGBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.vgg_models = VGGModels(self.configer)

    def __call__(self, vgg_cfg=None):
        arch = self.configer.get('network', 'backbone')
        if 'bn' in arch:
            arch_net = self.vgg_models.vgg_bn(vgg_cfg=vgg_cfg)

        else:
            arch_net = self.vgg_models.vgg(vgg_cfg=vgg_cfg)

        return arch_net
