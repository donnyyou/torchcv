#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch
import torch.nn as nn
import torch.nn.functional as F


class MixupSoftCELoss(nn.Module):
    def __init__(self, configer):
        super(MixupSoftCELoss, self).__init__()
        self.params_dict = dict()
        if 'soft_ce_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['mixup_soft_ce_loss']

        self.label_smooth = self.params_dict['label_smooth'] if 'label_smooth' in self.params_dict else 0.0
        assert 0 <= self.label_smooth <= 1
        reduction = self.params_dict['reduction'] if 'reduction' in self.params_dict else 'batchmean'
        self.criterion = nn.KLDivLoss(reduction=reduction)

    def forward(self, input, target_a, target_b, num_class, beta):
        one_hot_a = torch.zeros_like(input)
        one_hot_a.fill_(self.label_smooth * 1.0 / (num_class - 1))
        one_hot_a.scatter_(1, target_a.unsqueeze(1).long(), 1 - self.label_smooth)
        one_hot_b = torch.zeros_like(input)
        one_hot_b.fill_(self.label_smooth * 1.0 / (num_class - 1))
        one_hot_b.scatter_(1, target_b.unsqueeze(1).long(), 1 - self.label_smooth)
        input = F.log_softmax(input, 1)
        return beta * self.criterion(input, one_hot_a) + (1 - beta) * self.criterions(input, one_hot_b)
