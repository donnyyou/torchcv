#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# Select Cls Model for pose detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.cls.img.mlp import MLP
from models.cls.img.vgg import VGG19
from models.cls.img.shufflenetv2 import ShuffleNetV2, ShuffleResNetV2
from models.cls.img.mobilenet import MobileNet
from utils.tools.logger import Logger as Log


CLS_MODEL_DICT = {
    'mlp': MLP,
    'vgg19': VGG19,
    'mobilenet': MobileNet,
    'shufflenetv2': ShuffleNetV2,
    'shufflenetv2-50': ShuffleResNetV2,
    'shufflenetv2-164': ShuffleResNetV2
}


class ClsModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def image_classifier(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in CLS_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = CLS_MODEL_DICT[model_name](self.configer)

        return model
