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
        weight = self.configer.get('loss.params.ce_loss.weight', default=None)
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = self.configer.get('loss.params.ce_loss.reduction', default='mean')
        self.ignore_index = self.configer.get('loss.params.ce_loss.ignore_index', default=-100)

    def forward(self, input, target):
        target = self._scale_target(target, (input.size(2), input.size(3)))
        loss = F.cross_entropy(input, target,
                               weight=self.weight.to(input.device) if self.weight is not None else None,
                               ignore_index=self.ignore_index, reduction=self.reduction)
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()
