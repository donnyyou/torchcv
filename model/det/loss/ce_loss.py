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
        input = input.view(-1, input.size(-1))
        target = target.view(-1)
        loss = F.cross_entropy(input, target,
                               weight=self.weight.to(input.device) if self.weight is not None else None,
                               ignore_index=self.ignore_index, reduction=self.reduction)
        return loss

