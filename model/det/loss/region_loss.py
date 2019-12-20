#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import math
import numpy as np
import torch
import torch.nn as nn

from lib.tools.helper.det_helper import DetHelper


class RegionLoss(nn.Module):
    def __init__(self, configer):
        super(RegionLoss, self).__init__()
        self.configer = configer
        self.mse_loss = nn.MSELoss(reduction='mean')  # 'sum'
        self.bce_loss = nn.BCELoss(reduction='mean')

    def build_targets(self, detections, feat_list, data_dict):
        gt_bboxes = data_dict['bboxes']
        gt_labels = data_dict['labels']
        input_size = [data_dict['img'].size(3), data_dict['img'].size(2)]
        batch_target_list = list()
        batch_objmask_list = list()
        batch_noobjmask_list = list()
        for i, ori_anchors in enumerate(self.configer.get('anchor', 'anchors_list')):
            in_h, in_w = feat_list[i].size()[2:]
            w_fm_stride, h_fm_stride = input_size[0] / in_w, input_size[1] / in_h
            anchors = [(a_w / w_fm_stride, a_h / h_fm_stride) for a_w, a_h in ori_anchors]
            batch_size = len(gt_bboxes)
            num_anchors = len(anchors)
            obj_mask = torch.zeros(batch_size, num_anchors, in_h, in_w)
            noobj_mask = torch.ones(batch_size, num_anchors, in_h, in_w)
            tx = torch.zeros(batch_size, num_anchors, in_h, in_w)
            ty = torch.zeros(batch_size, num_anchors, in_h, in_w)
            tw = torch.zeros(batch_size, num_anchors, in_h, in_w)
            th = torch.zeros(batch_size, num_anchors, in_h, in_w)
            tconf = torch.zeros(batch_size, num_anchors, in_h, in_w)
            tcls = torch.zeros(batch_size, num_anchors, in_h, in_w, self.configer.get('data', 'num_classes'))

            for b in range(batch_size):
                for t in range(gt_bboxes[b].size(0)):
                    # Convert to position relative to box
                    gx = (gt_bboxes[b][t, 0] + gt_bboxes[b][t, 2]) / (2.0 * input_size[0]) * in_w
                    gy = (gt_bboxes[b][t, 1] + gt_bboxes[b][t, 3]) / (2.0 * input_size[1]) * in_h
                    gw = (gt_bboxes[b][t, 2] - gt_bboxes[b][t, 0]) / input_size[0] * in_w
                    gh = (gt_bboxes[b][t, 3] - gt_bboxes[b][t, 1]) / input_size[1] * in_h
                    if gw * gh == 0 or gx >= in_w or gy >= in_h:
                        continue

                    # Get grid box indices
                    gi = int(gx)
                    gj = int(gy)
                    # Get shape of gt box
                    gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0)
                    # Get shape of anchor box
                    anchor_shapes = torch.from_numpy(np.concatenate((np.zeros((num_anchors, 2)), np.array(anchors)), 1)).float()
                    # Calculate iou between gt and anchor shapes
                    anch_ious = DetHelper.bbox_iou(gt_box, anchor_shapes)
                    # Where the overlap is larger than threshold set mask to zero (ignore)
                    noobj_mask[b, anch_ious[0] > self.configer.get('anchor', 'iou_threshold')] = 0
                    # Find the best matching anchor box
                    best_n = torch.argmax(anch_ious, dim=1)
                    if anch_ious[0, best_n] < self.configer.get('anchor', 'iou_threshold'):
                        continue

                    # Masks
                    obj_mask[b, best_n, gj, gi] = 1
                    # Coordinates
                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    # Width and height
                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
                    # object
                    tconf[b, best_n, gj, gi] = 1
                    # One-hot encoding of label
                    tcls[b, best_n, gj, gi, int(gt_labels[b][t])] = 1

            obj_mask = obj_mask.view(batch_size, -1)
            noobj_mask = noobj_mask.view(batch_size, -1)
            tx = tx.view(batch_size, -1).unsqueeze(2)
            ty = ty.view(batch_size, -1).unsqueeze(2)
            tw = tw.view(batch_size, -1).unsqueeze(2)
            th = th.view(batch_size, -1).unsqueeze(2)
            tconf = tconf.view(batch_size, -1).unsqueeze(2)
            tcls = tcls.view(batch_size, -1, self.configer.get('data', 'num_classes'))
            target = torch.cat((tx, ty, tw, th, tconf, tcls), -1)
            batch_target_list.append(target)
            batch_objmask_list.append(obj_mask)
            batch_noobjmask_list.append(noobj_mask)

        batch_target = torch.cat(batch_target_list, 1)
        batch_objmask = torch.cat(batch_objmask_list, 1)
        batch_noobjmask = torch.cat(batch_noobjmask_list, 1)
        target_device = detections.device
        return batch_target.to(target_device), batch_objmask.to(target_device), batch_noobjmask.to(target_device)

    def forward(self, prediction, detections, targets, objmask, noobjmask):
        # targets, objmask, noobjmask = self.build_targets(detections, feat_list, data_dict)
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
        if objmask.sum() > 0:
            loss_x = self.bce_loss(x[objmask.byte()], tx[objmask.byte()])
            loss_y = self.bce_loss(y[objmask.byte()], ty[objmask.byte()])
            loss_w = self.mse_loss(w[objmask.byte()], tw[objmask.byte()])
            loss_h = self.mse_loss(h[objmask.byte()], th[objmask.byte()])
            loss_coord = loss_x + loss_y + loss_w + loss_h
            loss_obj = self.bce_loss(conf[objmask.byte()], objmask[objmask.byte()])
            loss_noobj = self.bce_loss(conf[noobjmask.byte()], objmask[noobjmask.byte()])
            loss_cls = self.bce_loss(pred_cls[objmask.byte()], tcls[objmask.byte()])
        else:
            zero = torch.tensor([0.0]).to(prediction.device)
            loss_coord, loss_obj, loss_noobj, loss_cls = zero, zero, zero, zero

        #  total loss = losses * weight
        loss = loss_coord * self.configer.get('loss', 'loss_weights')['coord_loss'] + \
               loss_obj * self.configer.get('loss', 'loss_weights')['obj_loss'] + \
               loss_noobj * self.configer.get('loss', 'loss_weights')['noobj_loss'] + \
               loss_cls * self.configer.get('loss', 'loss_weights')['cls_loss']

        return loss
