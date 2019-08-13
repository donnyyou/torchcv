#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    def __init__(self, configer):
        super(SmoothL1Loss, self).__init__()
        self.configer = configer

    def forward(self, pred_locs, gt_locs, gt_labels, sigma):
        pred_locs = pred_locs.contiguous().view(-1, pred_locs.size()[-1])
        gt_locs = gt_locs.contiguous().view(-1, gt_locs.size()[-1])
        in_weight = torch.zeros_like(gt_locs).to(gt_locs.device)
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike origin implementation,
        # we don't need inside_weight and outside_weight, they can calculate by gt_label
        in_weight[(gt_labels > 0).view(-1, 1).expand_as(in_weight)] = 1
        loc_loss = self.smooth_l1_loss(pred_locs, gt_locs, in_weight, sigma)
        # Normalize by total number of negtive and positive rois.
        loc_loss /= max((gt_labels.float() >= 0).sum().float(), 1e-9)  # ignore gt_label==-1 for rpn_loss
        return loc_loss

    @staticmethod
    def smooth_l1_loss(x, t, in_weight, sigma):
        sigma2 = sigma ** 2
        diff = in_weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.sum()

