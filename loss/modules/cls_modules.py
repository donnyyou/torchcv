#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# Loss function for Image Classification.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, configer=None):
        super(CrossEntropyLoss, self).__init__()
        super(CrossEntropyLoss, self).__init__()
        self.configer = configer
        weight = None
        if not self.configer.is_empty('cross_entropy_loss', 'weight'):
            weight = self.configer.get('cross_entropy_loss', 'weight')
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if not self.configer.is_empty('cross_entropy_loss', 'reduction'):
            reduction = self.configer.get("cross_entropy_loss", "reduction")

        ignore_index = -100
        if not self.configer.is_empty('cross_entropy_loss', 'ignore_index'):
            ignore_index = self.configer.get('cross_entropy_loss', 'ignore_index')

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, targets):
        return self.ce_loss(inputs, targets)
