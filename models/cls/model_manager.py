#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Cls Model for pose detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.cls.nets.mobilenet import MobileNet
from models.cls.nets.vgg import VGG
from models.cls.nets.shufflenetv2 import ShuffleNetV2, ShuffleResNetV2
from utils.tools.logger import Logger as Log

CLS_MODEL_DICT = {
    'vgg11': VGG,
    'vgg13': VGG,
    'vgg16': VGG,
    'vgg19': VGG,
    'mobilenet': MobileNet,
    'shufflenetv2': ShuffleNetV2,
    'shufflenetv2-50': ShuffleResNetV2,
    'shufflenetv2-164': ShuffleResNetV2
}


class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def image_classifier(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in CLS_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = CLS_MODEL_DICT[model_name](self.configer)

        return model
