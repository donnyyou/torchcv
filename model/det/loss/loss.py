#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch.nn as nn

from model.det.loss.focal_loss import FocalLoss
from model.det.loss.multibox_loss import MultiBoxLoss
from model.det.loss.smooth_l1_loss import SmoothL1Loss
from model.det.loss.ce_loss import CELoss
from model.det.loss.region_loss import RegionLoss


BASE_LOSS_DICT = dict(
    multibox_loss=0,
    smooth_l1_loss=1,
    ce_loss=2,
    focal_loss=2,
    region_loss=3,
)


class Loss(nn.Module):
    def __init__(self, configer):
        super(Loss, self).__init__()
        self.configer = configer
        self.func_list = [MultiBoxLoss(self.configer), SmoothL1Loss(self.configer), CELoss(self.configer),
                          FocalLoss(self.configer), RegionLoss(self.configer)]

    def forward(self, out_list):
        loss_dict = out_list[-1]
        out_dict = dict()
        weight_dict = dict()
        for key, item in loss_dict.items():
            out_dict[key] = self.func_list[int(item['type'].float().mean().item())](*item['params'])
            weight_dict[key] = item['weight'].mean().item()

        loss = 0.0
        for key in out_dict:
            loss += out_dict[key] * weight_dict[key]

        out_dict['loss'] = loss
        return out_dict
