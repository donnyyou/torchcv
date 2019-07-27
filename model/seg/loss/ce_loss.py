#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, configer=None):
        super(CELoss, self).__init__()
        self.configer = configer
        weight = self.configer.get('loss.params.ce_weight', default=None)
        weight = torch.FloatTensor(weight).cuda() if weight is not None else weight
        reduction = self.configer.get('loss.params.ce_reduction', default='mean')
        ignore_index = self.configer.get('loss.params.ce_ignore_index', default=-100)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, input, target):
        target = self._scale_target(target, (input.size(2), input.size(3)))
        loss = self.ce_loss(input, target)
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()
