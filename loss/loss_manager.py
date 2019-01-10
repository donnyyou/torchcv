#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss Manager for Image Classification.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from loss.modules.cls_modules import FCCELoss, FCCenterLoss
from loss.modules.det_modules import FRLoss, SSDMultiBoxLoss, YOLOv3Loss, SSDFocalLoss
from loss.modules.pose_modules import OPMseLoss
from loss.modules.seg_modules import FSCELoss, FSOhemCELoss, FSAuxCELoss, FSAuxEncCELoss, FSAuxOhemCELoss
from utils.tools.logger import Logger as Log


CLS_LOSS_DICT = {
    'fc_ce_loss': FCCELoss,
    'fc_center_loss': FCCenterLoss
}

DET_LOSS_DICT = {
    'ssd_multibox_loss': SSDMultiBoxLoss,
    'ssd_focal_loss': SSDFocalLoss,
    'yolov3_det_loss': YOLOv3Loss,
    'fr_loss': FRLoss
}

POSE_LOSS_DICT = {
    'op_mse_loss': OPMseLoss,
}

SEG_LOSS_DICT = {
    'fs_ce_loss': FSCELoss,
    'fs_ohemce_loss': FSOhemCELoss,
    'fs_auxce_loss':FSAuxCELoss,
    'fs_auxencce_loss': FSAuxEncCELoss,
    'fs_auxohemce_loss': FSAuxOhemCELoss
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _parallel(self, loss):
        if self.configer.get('network', 'loss_balance') and len(range(torch.cuda.device_count())) > 1:
            from extensions.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss

    def get_cls_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in CLS_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = CLS_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

    def get_seg_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

    def get_det_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in DET_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = DET_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

    def get_pose_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in POSE_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = POSE_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

