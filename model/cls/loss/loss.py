#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch.nn as nn

from model.cls.loss import KLLoss, CELoss, HardTripletLoss, LiftedStructureLoss, SoftCELoss, MixupCELoss, MixupSoftCELoss


BASE_LOSS_DICT = dict(
    ce_loss=0,
    kl_loss=1,
    hard_triplet_loss=2,
    lifted_structure_loss=3,
    soft_ce_loss=4,
    mixup_ce_loss=5,
    mixup_soft_ce_loss=6
)


class Loss(nn.Module):
    def __init__(self, configer):
        super(Loss, self).__init__()
        self.configer = configer
        self.func_list = [CELoss(self.configer), KLLoss(self.configer),
                          HardTripletLoss(self.configer), LiftedStructureLoss(self.configer),
                          SoftCELoss(self.configer), MixupCELoss(self.configer),
                          MixupSoftCELoss(self.configer)]

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
