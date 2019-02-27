#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import math
import numpy as np
import torch

from utils.helpers.det_helper import DetHelper


class YOLOTargetGenerator(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer):
        self.configer = configer

    def __call__(self, feat_list, batch_gt_bboxes, batch_gt_labels, input_size):
        batch_target_list = list()
        batch_objmask_list = list()
        batch_noobjmask_list = list()
        for i, ori_anchors in enumerate(self.configer.get('gt', 'anchors_list')):
            in_h, in_w = feat_list[i].size()[2:]
            w_fm_stride, h_fm_stride = input_size[0] / in_w, input_size[1] / in_h
            anchors = [(a_w / w_fm_stride, a_h / h_fm_stride) for a_w, a_h in ori_anchors]
            batch_size = len(batch_gt_bboxes)
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
                for t in range(batch_gt_bboxes[b].size(0)):
                    # Convert to position relative to box
                    gx = (batch_gt_bboxes[b][t, 0] + batch_gt_bboxes[b][t, 2]) / (2.0 * input_size[0]) * in_w
                    gy = (batch_gt_bboxes[b][t, 1] + batch_gt_bboxes[b][t, 3]) / (2.0 * input_size[1]) * in_h
                    gw = (batch_gt_bboxes[b][t, 2] - batch_gt_bboxes[b][t, 0]) / input_size[0] * in_w
                    gh = (batch_gt_bboxes[b][t, 3] - batch_gt_bboxes[b][t, 1]) /input_size[1] * in_h
                    if gw * gh == 0 or gx >= in_w or gy >= in_h:
                        continue

                    # Get grid box indices
                    gi = int(gx)
                    gj = int(gy)
                    # Get shape of gt box
                    gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                    # Get shape of anchor box
                    anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((num_anchors, 2)),
                                                                      np.array(anchors)), 1))
                    # Calculate iou between gt and anchor shapes
                    anch_ious = DetHelper.bbox_iou(gt_box, anchor_shapes)
                    # Where the overlap is larger than threshold set mask to zero (ignore)
                    noobj_mask[b, anch_ious[0] > self.configer.get('gt', 'iou_threshold')] = 0
                    # Find the best matching anchor box
                    best_n = torch.argmax(anch_ious, dim=1)
                    if anch_ious[0, best_n] < self.configer.get('gt', 'iou_threshold'):
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
                    tcls[b, best_n, gj, gi, int(batch_gt_labels[b][t])] = 1

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

        return batch_target, batch_objmask, batch_noobjmask
