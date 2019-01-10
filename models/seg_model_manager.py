#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Seg Model for semantic segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.seg.nets.denseassp import DenseASPP
from models.seg.nets.deeplabv3 import DeepLabV3
from models.seg.nets.embednet import EmbedNet
from models.seg.nets.pspnet import PSPNet
from utils.tools.logger import Logger as Log

SEG_MODEL_DICT = {
    'deeplabv3': DeepLabV3,
    'pspnet': PSPNet,
    'embednet': EmbedNet,
    'denseaspp': DenseASPP
}


class SegModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model
