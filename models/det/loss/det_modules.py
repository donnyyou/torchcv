#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.tools.logger import Logger as Log


class SSDFocalLoss(nn.Module):
    def __init__(self, configer):
        super(SSDFocalLoss, self).__init__()
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

    def forward(self, outputs, *targets, **kwargs):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds(tensor): predicted locations, sized [batch_size, #anchors, 4].
          loc_targets(tensor): encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds(tensor): predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets(tensor): encoded target labels, sized [batch_size, #anchors].
        Returns:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).

        """
        _, loc_preds, cls_preds = outputs
        loc_targets, cls_targets = targets

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


class SSDMultiBoxLoss(nn.Module):

    def __init__(self, configer):
        super(SSDMultiBoxLoss, self).__init__()
        self.num_classes = configer.get('data', 'num_classes')

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

    def forward(self, outputs, *targets, **kwargs):
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
        _, loc_preds, conf_preds = outputs
        loc_targets, conf_targets = targets
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


class YOLOv3Loss(nn.Module):
    def __init__(self, configer):
        super(YOLOv3Loss, self).__init__()
        self.configer = configer
        self.mse_loss = nn.MSELoss(reduction='sum')  # 'sum'
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, prediction, targets, objmask, noobjmask):
        # Get outputs
        x = prediction[..., 0]  # Center x
        y = prediction[..., 1]  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = prediction[..., 4]  # Conf
        pred_cls = prediction[..., 5:]  # Cls pred.

        # Get targets
        tx = targets[..., 0]  # Center x
        ty = targets[..., 1]  # Center y
        tw = targets[..., 2]  # Width
        th = targets[..., 3]  # Height
        tcls = targets[..., 5:]  # Cls pred.

        #  losses.
        obj_cnt = max(objmask.sum(), 1.0)
        loss_x = self.bce_loss(x * objmask, tx * objmask)
        loss_y = self.bce_loss(y * objmask, ty * objmask)
        loss_w = self.mse_loss(w * objmask, tw * objmask)
        loss_h = self.mse_loss(h * objmask, th * objmask)
        loss_coord = (loss_x + loss_y + 0.5 * loss_w + 0.5 * loss_h) / obj_cnt
        loss_hasobj = self.bce_loss(conf * objmask, objmask)
        loss_noobj = self.bce_loss(conf * noobjmask, noobjmask * 0.0)
        loss_obj = (loss_hasobj + 0.2 * loss_noobj) / obj_cnt
        loss_cls = self.bce_loss(pred_cls * objmask.unsqueeze(2), tcls * objmask.unsqueeze(2))
        loss_cls = loss_cls / obj_cnt

        #  total loss = losses * weight
        loss = loss_coord * self.configer.get('loss', 'loss_weights')['coord_loss'] + \
               loss_obj * self.configer.get('loss', 'loss_weights')['obj_loss'] + \
               loss_cls * self.configer.get('loss', 'loss_weights')['cls_loss']

        return loss


class FRLocLoss(nn.Module):
    def __init__(self, configer):
        super(FRLocLoss, self).__init__()
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


class FRLoss(nn.Module):

    def __init__(self, configer):
        super(FRLoss, self).__init__()
        self.configer = configer
        self.fr_loc_loss = FRLocLoss(configer)

    def forward(self, pred_group, target_group):
        # output_list: rpn_locs, rpn_scores, roi_cls_locs, roi_scores
        pred_rpn_locs, pred_rpn_scores, pred_roi_cls_locs, pred_roi_scores = pred_group
        gt_rpn_locs, gt_rpn_labels, gt_roi_cls_locs, gt_roi_labels = target_group
        gt_rpn_labels = gt_rpn_labels.contiguous().view(-1)
        pred_rpn_scores = pred_rpn_scores.contiguous().view(-1, 2)
        rpn_loc_loss = self.fr_loc_loss(pred_rpn_locs, gt_rpn_locs,
                                        gt_rpn_labels, self.configer.get('loss', 'params')['rpn_sigma'])

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(pred_rpn_scores, gt_rpn_labels, ignore_index=-1)

        roi_loc_loss = self.fr_loc_loss(pred_roi_cls_locs, gt_roi_cls_locs,
                                        gt_roi_labels, self.configer.get('loss', 'params')['roi_sigma'])
        roi_cls_loss = F.cross_entropy(pred_roi_scores, gt_roi_labels, ignore_index=-1)
        # Log.info('rpn loc {}'.format(rpn_loc_loss))
        # Log.info('rpn cls {}'.format(rpn_cls_loss))
        # Log.info('roi loc {}'.format(roi_loc_loss))
        # Log.info('roi cls {}'.format(roi_cls_loss))
        rpn_loss = (rpn_loc_loss + rpn_cls_loss) * self.configer.get('loss', 'loss_weights')['rpn_loss']
        roi_loss = (roi_loc_loss + roi_cls_loss) * self.configer.get('loss', 'loss_weights')['roi_loss']
        return rpn_loss + roi_loss

