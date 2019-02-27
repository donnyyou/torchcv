#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from models.backbones.vgg.vgg_backbone import VGGBackbone
from models.backbones.darknet.darknet_backbone import DarkNetBackbone
from models.backbones.resnet.resnet_backbone import ResNetBackbone
from models.backbones.mobilenet.mobilenet_backbone import MobileNetBackbone
from models.backbones.densenet.densenet_backbone import DenseNetBackbone
from models.backbones.squeezenet.squeezenet_backbone import SqueezeNetBackbone
from utils.tools.logger import Logger as Log


class BackboneSelector(object):

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.get('network', 'backbone')

        model = None
        if 'vgg' in backbone:
            model = VGGBackbone(self.configer)(**params)

        elif 'darknet' in backbone:
            model = DarkNetBackbone(self.configer)(**params)

        elif 'resnet' in backbone:
            model = ResNetBackbone(self.configer)(**params)

        elif 'mobilenet' in backbone:
            model = MobileNetBackbone(self.configer)(*params)

        elif 'densenet' in backbone:
            model = DenseNetBackbone(self.configer)(**params)

        elif 'squeezenet' in backbone:
            model = SqueezeNetBackbone(self.configer)(**params)

        else:
            Log.error('Backbone {} is invalid.'.format(backbone))
            exit(1)

        return model
