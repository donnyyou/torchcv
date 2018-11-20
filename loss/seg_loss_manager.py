#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# Loss Manager for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loss.modules.seg_modules import CrossEntropyLoss, FocalLoss, FCNSegLoss
from utils.tools.logger import Logger as Log


SEG_LOSS_DICT = {
    'cross_entropy_loss': CrossEntropyLoss,
    'focal_loss': FocalLoss,
    'fcn_seg_loss': FCNSegLoss
}


class SegLossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def get_seg_loss(self, key):
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = SEG_LOSS_DICT[key](self.configer)

        return loss
