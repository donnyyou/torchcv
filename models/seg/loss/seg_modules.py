#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


import torch
import torch.nn as nn
import torch.nn.functional as F


class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -100
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSOhemCELoss(nn.Module):
    def __init__(self, configer):
        super(FSOhemCELoss, self).__init__()
        self.configer = configer
        self.thresh = self.configer.get('loss', 'params')['ohem_thresh']
        self.min_kept = max(1, self.configer.get('loss', 'params')['ohem_minkeep'])
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        self.reduction = 'mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            self.reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -100
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        self.ignore_label = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, predict, target, **kwargs):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != self.ignore_label
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, self.thresh)
        loss_matirx = self.ce_loss(predict, target).contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


class FSAuxOhemCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxOhemCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.ohem_ce_loss = FSOhemCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ohem_ce_loss(seg_out, targets)
        aux_targets = self._scale_target(targets, (aux_out.size(2), aux_out.size(3)))
        aux_loss = self.ce_loss(aux_out, aux_targets)
        loss = self.configer.get('loss', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('loss', 'loss_weights')['aux_loss'] * aux_loss
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSAuxCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_targets = self._scale_target(targets, (aux_out.size(2), aux_out.size(3)))
        aux_loss = self.ce_loss(aux_out, aux_targets)
        loss = self.configer.get('loss', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('loss', 'loss_weights')['aux_loss'] * aux_loss
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSAuxEncCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxEncCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.se_loss = FSEncLoss(self.configer)

    def forward(self, outputs, targets, **kwargs):
        aux_out, enc_out, seg_out = outputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_targets = self._scale_target(targets, (aux_out.size(2), aux_out.size(3)))
        aux_loss = self.ce_loss(aux_out, aux_targets)
        loss = self.configer.get('loss', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('loss', 'loss_weights')['aux_loss'] * aux_loss
        enc_loss = self.enc_loss(enc_out, aux_targets, self.configer.get('loss', 'enc_size'))
        loss = loss + self.configer.get('loss', 'loss_weights')['enc_loss'] * enc_loss
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSFocalLoss(nn.Module):
    def __init__(self, configer):
        super(FSFocalLoss, self).__init__()
        self.configer = configer

    def forward(self, output, target, **kwargs):
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


class FSEncLoss(nn.Module):
    def __init__(self, configer):
        super(FSEncLoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'enc_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['enc_weight']
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'enc_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['enc_reduction']

        self.bce_loss = nn.BCELoss(weight, reduction=reduction)

    def forward(self, preds, targets, grid_size=None, **kwargs):
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


class FSEmbedLoss(nn.Module):
    def __init__(self, configer):
        super(FSEmbedLoss, self).__init__()
        self.num_classes = configer.get('data', 'num_classes')
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, inputs, targets, **kwargs):
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


if __name__ == "__main__":
    inputs = torch.ones((3, 5, 6, 6)).cuda()
    targets = torch.ones((3, 6, 6)).cuda()
