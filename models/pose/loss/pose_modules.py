#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Pose Estimation.


import torch
import torch.nn as nn
from torch.autograd import Variable


class OPMseLoss(nn.Module):
    def __init__(self, configer):
        super(OPMseLoss, self).__init__()
        self.configer = configer
        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'mse_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['mse_reduction']

        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, inputs, *targets, mask=None, weights=None):
        loss = 0.0
        if isinstance(inputs, list):
            if weights is not None:
                for i in range(len(inputs)):
                    if mask is not None:
                        loss += weights[i] * self.mse_loss(inputs[i]*mask, targets)
                    else:
                        loss += weights[i] * self.mse_loss(inputs[i], targets)
            else:
                for i in range(len(inputs)):
                    if mask is not None:
                        loss += self.mse_loss(inputs[i]*mask, targets)
                    else:
                        loss += self.mse_loss(inputs[i], targets)

        else:
            if mask is not None:
                loss = self.mse_loss(inputs*mask, targets)
            else:
                loss = self.mse_loss(inputs, targets)

        if self.configer.get('mse_loss', 'reduction') == 'sum':
            loss = loss / targets.size(0)

        return loss


class PartLoss(nn.Module):
    def __init__(self, configer):
        super(PartLoss, self).__init__()
        self.configer = configer
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.view(inputs.size(0), -1, 6, inputs.size(2), inputs.size(3))
        targets = targets.view(targets.size(0), -1, 6, targets.size(2), targets.size(3))
        paf_loss = self.mse_loss(inputs[:, :, 0:2, :, :], targets[:, :, 0:2, :, :])
        part_loss = self.mse_loss(inputs[:, :, 2:6, :, :], targets[:, :, 2:6, :, :])
        loss = paf_loss + part_loss * 6.0
        loss = loss / targets.size(0)
        return loss


class CapsuleLoss(nn.Module):

    def __init__(self, configer):
        super(CapsuleLoss, self).__init__()
        self.configer = configer
        self.mse_loss = nn.MSELoss(reduction=self.configer.get('capsule_loss', 'reduction'))

    def forward(self, inputs, targets, masks=None, is_focal=False):
        preds = torch.sqrt((inputs**2).sum(dim=1, keepdim=False))
        if masks is not None:
            preds = preds * masks

        if is_focal:
            loss = self.mse_loss(preds, targets)
        else:
            diff = preds - targets
            diff = diff ** 2
            alpha = 2.0
            weights = targets * alpha
            weights = torch.exp(weights)
            diff = weights * diff
            loss = diff.mean()

        return loss


class EmbeddingLoss(nn.Module):

    def __init__(self, configer):
        super(EmbeddingLoss, self).__init__()
        self.configer = configer
        self.num_keypoints = self.configer.get('data', 'num_keypoints')
        self.l_vec = self.configer.get('capsule', 'l_vec')
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, inputs, tags, numH, sigma=0.1):
        batch_size = inputs.size(0)
        h_tag_means = [[Variable(torch.zeros(self.l_vec,), requires_grad=True).cuda()
                        for h in range(numH[b].numpy()[0])] for b in range(inputs.size()[0])]

        for b in range(batch_size):
            for n in range(numH[b].numpy()[0]):
                valik = 0
                for k in range(self.num_keypoints):
                    tag = inputs[b].masked_select(tags[b][k].eq(n+1).unsqueeze(0))
                    if tag.size() != torch.Size([]):
                        h_tag_means[b][n] += tag
                        valik = valik + 1

                h_tag_means[b][n] = h_tag_means[b][n] / max(valik, 1)

        loss_list = list()
        for b in range(batch_size):
            for n in range(numH[b].numpy()[0]):
                for k in range(self.num_keypoints):
                    tag = inputs[b].masked_select(tags[b][k].eq(n+1).unsqueeze(0))
                    if tag.size() != torch.Size([]):
                        loss_list.append(self.mse_loss(tag, h_tag_means[b][n]))

        for b in range(batch_size):
            for n1 in range(numH[b].numpy()[0]):
                for n2 in range(numH[b].numpy()[0]):
                    if n1 != n2:
                        loss_same = torch.exp(-self.mse_loss(h_tag_means[b][n1], h_tag_means[b][n2]) / sigma / sigma)
                        loss_list.append(loss_same)

        if len(loss_list) == 0:
            loss = 0.0
        else:
            loss = loss_list[0]
            for i in range(len(loss_list)-1):
                loss += loss_list[i+1]

        return loss
