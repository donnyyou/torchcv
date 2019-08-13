#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Pose Estimation.


import torch.nn as nn


class MseLoss(nn.Module):
    def __init__(self, configer):
        super(MseLoss, self).__init__()
        self.configer = configer
        self.mse_loss = nn.MSELoss(reduction=self.configer.get('loss.params.mse_loss.reduction', default='mean'))

    def forward(self, pred, target):
        loss = self.mse_loss(pred, target)
        return loss
