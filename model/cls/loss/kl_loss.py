#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    def __init__(self, configer=None):
        super(KLLoss, self).__init__()
        self.params_dict = dict()
        if 'kl_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['kl_loss']

        reduction = self.params_dict['reduction'] if 'reduction' in self.params_dict else 'batchmean'
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, inputs, targets):

        return self.kl_loss(F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1))
