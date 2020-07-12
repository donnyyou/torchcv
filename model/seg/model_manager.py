#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Seg Model for semantic segmentation.


from model.seg.nets.denseassp import DenseASPP
from model.seg.nets.deeplabv3 import DeepLabV3
from model.seg.nets.pspnet import PSPNet
from model.seg.nets.sfnet import ResSFNet
from model.seg.nets.annn import asymmetric_non_local_network
from model.seg.loss.loss import Loss
from lib.tools.util.logger import Logger as Log


SEG_MODEL_DICT = {
    'deeplabv3': DeepLabV3,
    'pspnet': PSPNet,
    'denseaspp': DenseASPP,
    'annn': asymmetric_non_local_network,
    'res_sfnet': ResSFNet,
}


class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def get_seg_model(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model

    def get_seg_loss(self):
        if self.configer.get('network', 'gather'):
            return Loss(self.configer)

        from lib.parallel.data_parallel import ParallelCriterion
        return ParallelCriterion(Loss(self.configer))
