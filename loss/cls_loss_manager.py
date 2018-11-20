#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# Loss Manager for Image Classification.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loss.modules.cls_modules import CrossEntropyLoss
from utils.tools.logger import Logger as Log


CLS_LOSS_DICT = {
    'cross_entropy_loss': CrossEntropyLoss,
}


class ClsLossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def get_cls_loss(self, key):
        if key not in CLS_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = CLS_LOSS_DICT[key](self.configer)

        return loss
