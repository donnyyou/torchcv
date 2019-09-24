#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# SqueezeNet models.


from model.backbone.squeezenet.squeezenet_models import SqueezeNetModels


class SqueezeNetBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.squeeze_models = SqueezeNetModels(self.configer)

    def __call__(self, backbone=None, pretrained=None):
        arch = self.configer.get('network.backbone') if backbone is None else backbone
        pretrained = self.configer.get('network.pretrained') if pretrained is None else pretrained
        if arch == 'squeezenet':
            arch_net = self.squeeze_models.squeezenet()

        elif arch == 'squeezenet_dilated8':
            arch_net = self.squeeze_models.squeezenet_dilated8()

        elif arch == 'squeezenet_dilated16':
            arch_net = self.squeeze_models.squeezenet()

        else:
            raise Exception('Architecture undefined!')

        return arch_net
