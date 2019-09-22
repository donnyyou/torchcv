#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch
import torch.nn as nn


class MixupCELoss(nn.Module):
    def __init__(self, configer):
        super(MixupCELoss, self).__init__()
        self.params_dict = dict()
        if 'ce_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['mixup_ce_loss']

        weight = torch.FloatTensor(self.params_dict['weight']).cuda() if 'weight' in self.params_dict else None
        reduction = self.params_dict['reduction'] if 'reduction' in self.params_dict else 'mean'
        ignore_index = self.params_dict['ignore_index'] if 'ignore_index' in self.params_dict else -100

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, input, target_a, target_b, beta):

        return beta * self.ce_loss(input, target_a) + (1 - beta) * self.ce_loss(input, target_b)
