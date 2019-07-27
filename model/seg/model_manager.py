#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Seg Model for semantic segmentation.


import torch

from model.seg.nets.denseassp import DenseASPP
from model.seg.nets.deeplabv3 import DeepLabV3
from model.seg.nets.pspnet import PSPNet
from model.seg.loss.seg_modules import SegCELoss, SegOhemCELoss, SegAuxCELoss, SegAuxEncCELoss, SegAuxOhemCELoss
from tools.util.logger import Logger as Log


SEG_MODEL_DICT = {
    'deeplabv3': DeepLabV3,
    'pspnet': PSPNet,
    'denseaspp': DenseASPP
}

SEG_LOSS_DICT = {
    'seg_ce_loss': SegCELoss,
    'seg_ohemce_loss': SegOhemCELoss,
    'seg_auxce_loss':SegAuxCELoss,
    'seg_auxencce_loss': SegAuxEncCELoss,
    'seg_auxohemce_loss': SegAuxOhemCELoss
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

    def get_seg_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = SEG_LOSS_DICT[key](self.configer)
        if self.configer.get('network', 'loss_balance') and len(range(torch.cuda.device_count())) > 1:
            from exts.tools.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss
