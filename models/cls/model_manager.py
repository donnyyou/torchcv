#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Cls Model for pose detection.


import torch

from models.cls.nets.mobilenet import MobileNet
from models.cls.nets.vgg import VGG
from models.cls.nets.shufflenetv2 import ShuffleNetV2, ShuffleResNetV2
from models.cls.loss.cls_modules import ICCELoss, ICCenterLoss
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

CLS_LOSS_DICT = {
    'ic_ce_loss': ICCELoss,
    'ic_center_loss': ICCenterLoss
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

    def get_cls_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in CLS_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = CLS_LOSS_DICT[key](self.configer)
        if self.configer.get('network', 'loss_balance') and len(range(torch.cuda.device_count())) > 1:
            from extensions.tools.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss
