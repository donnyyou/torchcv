#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Mobilenet models.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.backbones.mobilenet.mobilenet_models import MobileNetModels


class MobileNetBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.mobile_models = MobileNetModels(self.configer)

    def __call__(self):
        arch = self.configer.get('network', 'backbone')

        if arch == 'mobilenetv2':
            arch_net = self.mobile_models.mobilenetv2()

        elif arch == 'mobilenetv2_dilated8':
            arch_net = self.mobile_models.mobilenetv2_dilated8()

        else:
            raise Exception('Architecture undefined!')

        return arch_net