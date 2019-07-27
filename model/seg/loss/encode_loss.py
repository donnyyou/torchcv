#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


import torch
import torch.nn as nn
import torch.nn.functional as F


class EncodeLoss(nn.Module):
    def __init__(self, configer):
        super(EncodeLoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'enc_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['enc_weight']
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'mean'
        if self.configer.exists('loss', 'params') and 'enc_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['enc_reduction']

        self.bce_loss = nn.BCELoss(weight, reduction=reduction)
        self.grid_size = self.configer.get('loss', 'params')['enc_grid_size']

    def forward(self, preds, targets):
        if len(targets.size()) == 2:
            return self.bce_loss(F.sigmoid(preds), targets)

        targets = self._scale_target(targets, (preds.size(2), preds.size(3)))
        se_target = self._get_batch_label_vector(targets,
                                                 self.configer.get('data', 'num_classes'),
                                                 self.grid_size).type_as(preds)
        return self.bce_loss(F.sigmoid(preds), se_target)

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

    @staticmethod
    def _get_batch_label_vector(target_, num_classes, grid_size=None):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        b, h, w = target_.size()
        pad_h = 0 if (h % grid_size == 0) else grid_size - (h % grid_size)
        pad_w = 0 if (w % grid_size == 0) else grid_size - (w % grid_size)
        target = target_.clone()
        target = F.pad(target, (0, pad_w, 0, pad_h), "constant", num_classes)

        b, h, w = target.size()
        target = target.contiguous().view(b, h // grid_size, grid_size, w // grid_size, grid_size)
        target = target.permute(0, 1, 3, 2, 4).contiguous().view(b * h * w // (grid_size ** 2), grid_size, grid_size)

        batch = target.size(0)
        tvect = torch.zeros(batch, num_classes)
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=num_classes, min=0, max=num_classes - 1)
            vect = hist>0
            tvect[i] = vect

        return tvect
