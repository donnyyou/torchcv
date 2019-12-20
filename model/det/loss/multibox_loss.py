#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.tools.helper.det_helper import DetHelper
from model.det.layers.ssd_target_generator import SSDTargetGenerator
from lib.tools.util.logger import Logger as Log


class MultiBoxLoss(nn.Module):

    def __init__(self, configer):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = configer.get('data', 'num_classes')
        self.ssd_target_generator = SSDTargetGenerator(configer)

    def _cross_entropy_loss(self, x, y):
        """Cross entropy loss w/o averaging across all samples.

        Args:
          x(tensor): sized [N,D]
          y(tensor): sized [N,]

        Returns:
          (tensor): cross entropy loss, sized [N,]

        """
        xmax = x.data.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(x - xmax), dim=1)) + xmax
        return log_sum_exp.view(-1, 1) - x.gather(1, y.view(-1, 1))

    def _hard_negative_mining(self, conf_loss, pos):
        """Return negative indices that is 3x the number as positive indices.

        Args:
          conf_loss: (tensor) cross entropy loss between conf_preds and conf_targets, sized [N*8732,]
          pos: (tensor) positive(matched) box indices, sized [N, 8732]
        Returns:
          (tensor): negative indices, sized [N, 8732]

        """
        batch_size, num_boxes = pos.size()

        conf_loss = conf_loss.view(batch_size, -1)  # [N,8732]
        conf_loss[pos] = 0  # set pos boxes = 0, the rest are neg conf_loss

        _, idx = conf_loss.sort(1, descending=True)  # sort by neg conf_loss
        _, rank = idx.sort(1)  # [N,8732]

        num_pos = pos.long().sum(1)  # [N,1]
        num_neg = torch.clamp(3 * num_pos, min=1, max=num_boxes-1)  # [N,1]
        neg = rank < num_neg.unsqueeze(1).expand_as(rank)  # [N,8732]
        return neg

    @staticmethod
    def smooth_l1_loss(x, t):
        diff = (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < 1.).float()
        y = flag * (diff ** 2) * 0.5 + (1 - flag) * (abs_diff - 0.5)
        return y.sum()

    def forward(self, loc_preds, conf_preds, loc_targets, conf_targets):
        """Compute loss between (loc_preds, loc_targets) and (conf_preds, conf_targets).

        Args:
          loc_preds(tensor): predicted locations, sized [batch_size, 8732, 4]
          loc_targets(tensor): encoded target locations, sized [batch_size, 8732, 4]
          conf_preds(tensor): predicted class confidences, sized [batch_size, 8732, num_classes]
          conf_targets:(tensor): encoded target classes, sized [batch_size, 8732]
          is_print: whether print loss
          img: using for visualization

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(conf_preds, conf_targets)
          loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
          conf_loss = CrossEntropyLoss(pos_conf_preds, pos_conf_targets)
                    + CrossEntropyLoss(neg_conf_preds, neg_conf_targets)

        """
        # loc_targets, conf_targets = self.ssd_target_generator(feat_list, data_dict)
        batch_size, num_boxes, _ = loc_preds.size()

        pos = conf_targets > 0  # [N,8732], pos means the box matched.
        num_matched_boxes = pos.data.float().sum()
        if num_matched_boxes == 0:
            print("No matched boxes")

        # loc_loss.
        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N, 8732, 4]
        pos_loc_preds = loc_preds[pos_mask].view(-1, 4)  # [pos,4]
        pos_loc_targets = loc_targets[pos_mask].view(-1, 4)  # [pos,4]
        loc_loss = self.smooth_l1_loss(pos_loc_preds, pos_loc_targets)  # F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, reduction='sum')

        # conf_loss.
        conf_loss = self._cross_entropy_loss(conf_preds.view(-1, self.num_classes), conf_targets.view(-1))  # [N*8732,]
        neg = self._hard_negative_mining(conf_loss, pos)    # [N,8732]
        pos_mask = pos.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
        neg_mask = neg.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
        mask = (pos_mask + neg_mask).gt(0)
        pos_and_neg = (pos + neg).gt(0)
        preds = conf_preds[mask].view(-1, self.num_classes)  # [pos + neg,21]
        targets = conf_targets[pos_and_neg]                  # [pos + neg,]
        conf_loss = F.cross_entropy(preds, targets, reduction='sum', ignore_index=-1)

        if num_matched_boxes > 0:
            loc_loss = loc_loss / num_matched_boxes
            conf_loss = conf_loss / num_matched_boxes
        else:
            return conf_loss + loc_loss

        Log.debug("loc_loss: %f, cls_loss: %f" % (float(loc_loss.item()), float(conf_loss.item())))

        return loc_loss + conf_loss
