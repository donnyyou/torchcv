#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.tools.util.logger import Logger as Log


class FocalLoss(nn.Module):
    def __init__(self, configer):
        super(FocalLoss, self).__init__()
        self.num_classes = configer.get('data', 'num_classes')

    def _one_hot_embeding(self, labels):
        """Embeding labels to one-hot form.

        Args:
            labels(LongTensor): class labels
            num_classes(int): number of classes
        Returns:
            encoded labels, sized[N, #classes]

        """

        y = torch.eye(self.num_classes)  # [D, D]
        return y[labels]  # [N, D]

    def focal_loss(self, x, y):
        """Focal loss

        Args:
            x(tensor): size [N, D]
            y(tensor): size [N, ]
        Returns:
            (tensor): focal loss

        """

        alpha = 0.25
        gamma = 2

        t = self._one_hot_embeding(y.data.cpu())
        t = Variable(t).cuda()  # [N, 20]

        logit = F.softmax(x)
        logit = logit.clamp(1e-7, 1.-1e-7)
        conf_loss_tmp = -1 * t.float() * torch.log(logit)
        conf_loss_tmp = alpha * conf_loss_tmp * (1-logit)**gamma
        conf_loss = conf_loss_tmp.sum()

        return conf_loss

    def forward(self, loc_preds, cls_preds, loc_targets, cls_targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds(tensor): predicted locations, sized [batch_size, #anchors, 4].
          loc_targets(tensor): encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds(tensor): predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets(tensor): encoded target labels, sized [batch_size, #anchors].
        Returns:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).

        """

        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction='sum')

        # cls_loss = FocalLoss(loc_preds, loc_targets)
        pos_neg = cls_targets > -1  # exclude ignored anchors
        # num_pos_neg = pos_neg.data.long().sum()
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        num_pos = max(1.0, num_pos)

        Log.debug('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0] / num_pos, cls_loss.data[0] / num_pos))

        loss = loc_loss / num_pos + cls_loss / num_pos

        return loss
