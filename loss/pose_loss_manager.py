#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# Loss function for Pose Estimation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loss.modules.pose_modules import CapsuleLoss, MarginLoss, MseLoss, PartLoss
from loss.modules.pose_modules import EmbeddingLoss
from loss.modules.pose_modules import VoteLoss
from utils.tools.logger import Logger as Log


POSE_LOSS_DICT = {
    'capsule_loss': CapsuleLoss,
    'margin_loss': MarginLoss,
    'mse_loss': MseLoss,
    'part_loss': PartLoss,
    'embedding_loss': EmbeddingLoss,
    'vote_loss': VoteLoss,
}


class PoseLossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def get_pose_loss(self, key):
        if key not in POSE_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = POSE_LOSS_DICT[key](self.configer)

        return loss
