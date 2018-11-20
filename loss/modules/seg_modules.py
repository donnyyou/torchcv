#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# Loss function for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, configer=None):
        super(CrossEntropyLoss, self).__init__()
        self.configer = configer
        weight = None
        if not self.configer.is_empty('cross_entropy_loss', 'weight'):
            weight = self.configer.get('cross_entropy_loss', 'weight')
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if not self.configer.is_empty('cross_entropy_loss', 'reduction'):
            reduction = self.configer.get("cross_entropy_loss", "reduction")

        ignore_index = -100
        if not self.configer.is_empty('cross_entropy_loss', 'ignore_index'):
            ignore_index = self.configer.get('cross_entropy_loss', 'ignore_index')

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, targets, weights=None):
        loss = 0.0
        if isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if isinstance(targets, list):
                    loss += weights[i] * self.ce_loss(inputs[i], targets[i])
                else:
                    loss += weights[i] * self.ce_loss(inputs[i], targets)

        else:
            loss = self.ce_loss(inputs, targets)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, configer):
        super(FocalLoss, self).__init__()
        self.configer = configer

    def forward(self, output, target):
        self.y = self.configer.get('focal_loss', 'y')
        P = F.softmax(output)
        f_out = F.log_softmax(output)
        Pt = P.gather(1, torch.unsqueeze(target, 1))
        focus_p = torch.pow(1 - Pt, self.y)
        alpha = 0.25
        nll_feature = -f_out.gather(1, torch.unsqueeze(target, 1))
        weight_nll = alpha * focus_p * nll_feature
        loss = weight_nll.mean()
        return loss


class SegEncodeLoss(nn.Module):
    def __init__(self, configer):
        super(SegEncodeLoss, self).__init__()
        self.configer = configer
        weight = None
        if not self.configer.is_empty('seg_encode_loss', 'weight'):
            weight = self.configer.get('seg_encode_loss', 'weight')
            weight = torch.FloatTensor(weight).cuda()

        size_average = True
        if not self.configer.is_empty('seg_encode_loss', 'size_average'):
            size_average = self.configer.get('seg_encode_loss', 'size_average')

        reduce = True
        if not self.configer.is_empty('seg_encode_loss', 'reduce'):
            reduce = self.configer.get("seg_encode_loss", "reduce")

        self.bce_loss = nn.BCELoss(weight, size_average, reduce=reduce)

    def forward(self, preds, targets, grid_size=None):
        if len(targets.size()) == 2:
            return self.bce_loss(F.sigmoid(preds), targets)

        se_target = self._get_batch_label_vector(targets,
                                                 self.configer.get('data', 'num_classes'),
                                                 grid_size).type_as(preds)
        return self.bce_loss(F.sigmoid(preds), se_target)

    @staticmethod
    def _get_batch_label_vector(target_, num_classes, grid_size=None):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        b, h, w = target_.size()
        pad_h = 0 if (h % grid_size == 0) else grid_size - (h % grid_size)
        pad_w = 0 if (w % grid_size == 0) else grid_size - (w % grid_size)
        target = target_.clone()
        target = F.pad(target, (0, pad_w, 0, pad_h), "constant", num_classes)

        b, h, w = target.size()
        if grid_size is not None:
            target = target.contiguous().view(b, h // grid_size, grid_size, w // grid_size, grid_size)
            target = target.permute(0, 1, 3, 2, 4).contiguous().view(b * h * w // (grid_size ** 2),
                                                                     grid_size, grid_size)

        batch = target.size(0)
        tvect = torch.zeros(batch, num_classes)
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=num_classes, min=0, max=num_classes - 1)
            vect = hist>0
            tvect[i] = vect

        return tvect


class EmbeddingLoss(nn.Module):
    def __init__(self, configer):
        super(EmbeddingLoss, self).__init__()
        self.num_classes = configer.get('data', 'num_classes')
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, inputs, targets):
        inputs = inputs.transpose(0, 1)
        center_array = torch.zeros((self.num_classes, inputs.size()[0]), requires_grad=True).cuda()
        sim_loss = torch.Tensor([0.0]).cuda()
        sim_loss.requires_grad = True

        mask_list = list()
        for i in range(self.num_classes):
            mask = self.get_mask(targets, i).unsqueeze(0)
            sum_pixel = max(mask.sum(), 1)
            # print sum_pixel
            mask = mask.contiguous().repeat(inputs.size()[0], 1, 1, 1).byte().cuda()
            sim_input = inputs[mask]
            if sim_input.numel() == 0:
                mask_list.append(i)
                continue

            sim_input = sim_input.contiguous().view(inputs.size()[0], -1)
            center = torch.sum(sim_input, 1, keepdim=False)
            center = center / sum_pixel
            center_array[i, :] = center

            sim_input = sim_input.permute(1, 0)

            sim_label = torch.ones(sim_input.size()[0], ).float().cuda()
            sim_center = center.contiguous().view(1, -1).repeat(sim_input.size()[0], 1)
            sim_loss = sim_loss + self.cosine_loss(sim_center, sim_input, sim_label)

        diff_loss = torch.Tensor([0.0]).cuda()
        diff_loss.requires_grad = True
        for i in range(self.num_classes):
            if i in mask_list:
                continue

            label = torch.zeros(self.num_classes, ).float().cuda()
            center_dual = torch.zeros((self.num_classes, inputs.size()[0]), requires_grad=True).cuda()
            for k in range(self.num_classes):
                center_dual[k] = center_array[i]

            for j in range(self.num_classes):
                if j == i:
                    label[j] = 1
                else:
                    label[j] = -1

            diff_loss = diff_loss + self.cosine_loss(center_array, center_dual, label)

        embedding_loss = diff_loss + sim_loss
        # print embedding_loss.requires_grad
        return embedding_loss

    def get_mask(self, targets, i):
        targets_cp = torch.cuda.FloatTensor(targets.size())
        targets_cp.copy_(targets.data)
        if i == 0:
            targets_cp[targets_cp != 0] = 2
            targets_cp[targets_cp == 0] = 1
            targets_cp[targets_cp == 2] = 0
        else:
            targets_cp[targets_cp != i] = 0
            targets_cp[targets_cp == i] = 1

        return targets_cp


class FCNSegLoss(nn.Module):
    def __init__(self, configer):
        super(FCNSegLoss, self).__init__()
        self.configer = configer
        self.ce_loss = CrossEntropyLoss(self.configer)
        self.se_loss = SegEncodeLoss(self.configer)
        self.focal_loss = FocalLoss(self.configer)
        self.embed_loss = EmbeddingLoss(self.configer)

    def forward(self, outputs, targets):
        if self.configer.get('network', 'model_name') == 'grid_encnet':
            seg_out, se_out, aux_out = outputs
            seg_loss = self.ce_loss(seg_out, targets)
            aux_targets = self._scale_target(targets, (aux_out.size(2), aux_out.size(3)))
            aux_loss = self.ce_loss(aux_out, aux_targets)
            loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
            loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
            se_loss = self.se_loss(se_out, aux_targets, self.configer.get('network', 'enc_size'))
            loss = loss + self.configer.get('network', 'loss_weights')['se_loss'] * se_loss

            return loss

        elif self.configer.get('network', 'model_name') == 'pspnet':
            seg_out, aux_out = outputs
            seg_loss = self.ce_loss(seg_out, targets)
            aux_targets = self._scale_target(targets, (aux_out.size(2), aux_out.size(3)))
            aux_loss = self.ce_loss(aux_out, aux_targets)
            loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
            loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
            return loss

        elif self.configer.get('network', 'model_name') == 'embednet':
            seg_out, aux_out, embed_out = outputs
            seg_loss = self.ce_loss(seg_out, targets)
            aux_targets = self._scale_target(targets, (aux_out.size(2), aux_out.size(3)))
            aux_loss = self.ce_loss(aux_out, aux_targets)
            loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
            loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
            embed_loss = self.embed_loss(embed_out, aux_targets)
            loss = loss + self.configer.get('network', 'loss_weights')['embed_loss'] * embed_loss
            return loss

        else:
            return self.ce_loss(outputs, targets)

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()





if __name__ == "__main__":
    inputs = torch.ones((3, 5, 6, 6)).cuda()
    targets = torch.ones((3, 6, 6)).cuda()
    embed_loss = EmbeddingLoss(2)
    print(embed_loss(inputs, targets))
