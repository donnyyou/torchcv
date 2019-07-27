#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch.nn as nn

from model.seg.loss.ce_loss import CELoss
from model.seg.loss.embed_loss import EmbedLoss
from model.seg.loss.ohem_ce_loss import OhemCELoss
from model.seg.loss.focal_ce_loss import FocalCELoss
from model.seg.loss.encode_loss import EncodeLoss


BASE_LOSS_DICT = dict(
    ce_loss=0,
    ohem_ce_loss=1,
    focal_ce_loss=2,
    embed_loss=3,
    encode_loss=4
)


class Loss(nn.Module):
    def __init__(self, configer):
        super(Loss, self).__init__()
        self.configer = configer
        self.func_list = [CELoss(self.configer), OhemCELoss(self.configer),
                          FocalCELoss(self.configer), EmbedLoss(self.configer),
                          EncodeLoss(self.configer)]

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
