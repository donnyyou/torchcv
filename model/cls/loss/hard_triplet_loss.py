#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch
import torch.nn as nn
import torch.nn.functional as F


class HardTripletLoss(nn.Module):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, configer=None):
        super(HardTripletLoss, self).__init__()
        self.params_dict = dict()
        if 'hard_triplet_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['hard_triplet_loss']
        self.margin = self.params_dict['margin'] if 'margin' in self.params_dict else None
        if self.margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    @staticmethod
    def normalize(x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
          x: pytorch Variable
        Returns:
          x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    @staticmethod
    def hard_example_mining(dist_mat, labels, return_inds=False):
        """For each anchor, find the hardest positive and negative sample.
        Args:
          dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
          labels: pytorch LongTensor, with shape [N]
          return_inds: whether to return the indices. Save time if `False`(?)
        Returns:
          dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
          dist_an: pytorch Variable, distance(anchor, negative); shape [N]
          p_inds: pytorch LongTensor, with shape [N];
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
          n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        NOTE: Only consider the case in which all labels have same num of samples,
          thus we can cope with all anchors in parallel.
        """

        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # shape [N, N]
        is_pos = torch.eq(*[labels.unsqueeze(dim).expand_as(dist_mat) for dim in [0, 1]])
        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, relative_p_inds = torch.max(dist_mat * is_pos.float(), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_mat_neg = dist_mat * (1 - is_pos).float()
        dist_mat_neg[is_pos] = dist_mat_neg.max().item() + 1.0
        dist_an, relative_n_inds = torch.min(dist_mat_neg, 1, keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        if return_inds:
            # shape [N, N]
            ind = (labels.new().resize_as_(labels)
                   .copy_(torch.arange(0, N).long())
                   .unsqueeze(0).expand(N, N))
            # shape [N, 1]
            p_inds = torch.gather(
                ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
            n_inds = torch.gather(
                ind[(1 - is_pos).byte()].contiguous().view(N, -1), 1, relative_n_inds.data)
            # shape [N]
            p_inds = p_inds.squeeze(1)
            n_inds = n_inds.squeeze(1)
            return dist_ap, dist_an, p_inds, n_inds

        return dist_ap, dist_an

    @staticmethod
    def pdist(A, squared=False):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=1e-12)
        return res if squared else res.sqrt()

    def forward(self, global_feat, labels):
        if 'normalize' in self.params_dict and self.params_dict['normalize']:
            global_feat = self.normalize(global_feat, axis=-1)

        dist_mat = self.pdist(global_feat, squared=False)
        dist_ap, dist_an = self.hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss  # , dist_ap, dist_an