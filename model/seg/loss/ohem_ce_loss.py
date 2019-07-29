#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


import torch
import torch.nn as nn


class OhemCELoss(nn.Module):
    def __init__(self, configer):
        super(OhemCELoss, self).__init__()
        self.configer = configer
        weight = self.configer.get('loss.params.ohem_ce_loss.weight', default=None)
        weight = torch.FloatTensor(weight) if weight is not None else weight
        reduction = self.configer.get('loss.params.ohem_ce_loss.reduction', default='mean')
        ignore_index = self.configer.get('loss.params.ohem_ce_loss.ignore_index', default=-100)
        self.thresh = self.configer.get('loss.params.ohem_ce_loss.thresh', default=0.7)
        self.min_kept = max(1, self.configer.get('loss.params.ohem_ce_loss.minkeep', default=1))
        self.ignore_label = ignore_index

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        target = self._scale_target(target, (predict.size(2), predict.size(3)))
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != self.ignore_label
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, self.thresh)
        loss_matrix = F.cross_entropy(predict, target, weight=self.weight.to(input.device),
                                      ignore_index=self.ignore_index, reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()
