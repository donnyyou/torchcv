#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Pose Estimation.


import torch.nn as nn


class MseLoss(nn.Module):
    def __init__(self, configer):
        super(MseLoss, self).__init__()
        self.configer = configer
        self.reduction = self.configer.get('loss.params.mse_loss.reduction', default='mean')
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, pred, target):
        loss = self.mse_loss(pred, target)
        return loss / pred.size(0) if self.reduction == 'sum' else loss
