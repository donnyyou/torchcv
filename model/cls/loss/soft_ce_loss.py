#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCELoss(nn.Module):
    def __init__(self, configer):
        super(SoftCELoss, self).__init__()
        self.params_dict = dict()
        if 'soft_ce_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['soft_ce_loss']

        self.label_smooth = self.params_dict['label_smooth'] if 'label_smooth' in self.params_dict else 0.0
        assert 0 <= self.label_smooth <= 1
        reduction = self.params_dict['reduction'] if 'reduction' in self.params_dict else 'batchmean'
        self.criterion = nn.KLDivLoss(reduction=reduction)

    def forward(self, input, target, num_class):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smooth * 1.0 / (num_class - 1))
        one_hot.scatter_(1, target.unsqueeze(1).long(), 1 - self.label_smooth)
        input = F.log_softmax(input, 1)
        return self.criterion(input, one_hot)
