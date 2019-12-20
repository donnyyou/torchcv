#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import numpy as np
import torch

from lib.tools.helper.det_helper import DetHelper
from model.det.layers.fr_priorbox_layer import FRPriorBoxLayer


class RPNTargetAssigner(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer):
        self.configer = configer
        self.fr_proirbox_layer = FRPriorBoxLayer(configer)

    def __call__(self, feat_list, gt_bboxes, meta):
        anchor_boxes = self.fr_proirbox_layer(feat_list, meta[0]['input_size'])
        n_sample = self.configer.get('rpn', 'sampler')['n_sample']
        pos_iou_thresh = self.configer.get('rpn', 'sampler')['pos_iou_thresh']
        neg_iou_thresh = self.configer.get('rpn', 'sampler')['neg_iou_thresh']
        pos_ratio = self.configer.get('rpn', 'sampler')['pos_ratio']
        # Calc indicies of anchors which are located completely inside of the image
        # whose size is speficied.
        target_bboxes = list()
        target_labels = list()
        device = feat_list[0].device
        anchor_boxes = anchor_boxes.to(device)
        for i in range(len(gt_bboxes)):
            index_inside = (((anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2) >= 0)
                            & ((anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2) >= 0)
                            & ((anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2) < meta[i]['border_wh'][0])
                            & ((anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2) < meta[i]['border_wh'][1]))
            index_inside = index_inside.nonzero().contiguous().view(-1, )
            default_boxes = anchor_boxes[index_inside]
            loc = torch.zeros_like(default_boxes)
            label = torch.ones((default_boxes.size(0),)).mul_(-1).long().to(device)

            if gt_bboxes[i].numel() > 0:
                # label: 1 is positive, 0 is negative, -1 is dont care
                ious = DetHelper.bbox_iou(gt_bboxes[i],
                                          torch.cat([default_boxes[:, :2] - default_boxes[:, 2:] / 2,
                                                     default_boxes[:, :2] + default_boxes[:, 2:] / 2], 1))

                max_ious, argmax_ious = ious.max(0, keepdim=False)
                _, gt_argmax_ious = ious.max(1, keepdim=False)

                # assign negative labels first so that positive labels can clobber them
                label[max_ious < neg_iou_thresh] = 0

                # positive label: for each gt, anchor with highest iou
                label[gt_argmax_ious] = 1

                # positive label: above threshold IOU
                label[max_ious >= pos_iou_thresh] = 1

                # subsample positive labels if we have too many
                n_pos = int(pos_ratio * n_sample)
                pos_index = (label == 1).nonzero().contiguous().view(-1, ).cpu().numpy()
                if len(pos_index) > n_pos:
                    disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
                    label[disable_index] = -1

                # subsample negative labels if we have too many
                n_neg = n_sample - torch.sum(label == 1).item()
                neg_index = (label == 0).nonzero().contiguous().view(-1, ).cpu().numpy()

                if len(neg_index) > n_neg:
                    disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
                    label[disable_index] = -1

                boxes = gt_bboxes[i][argmax_ious]  # [8732,4]
                cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]  # [8732,2]
                cxcy /= default_boxes[:, 2:]
                wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]  # [8732,2]
                wh = torch.log(wh)
                loc = torch.cat([cxcy, wh], 1)  # [8732,4]
                # loc = loc[:, [1, 0, 3, 2]]
            else:
                # subsample negative labels if we have too many
                n_neg = n_sample // 2
                neg_index = (label == -1).nonzero().contiguous().view(-1, ).numpy()
                if len(neg_index) > n_neg:
                    disable_index = np.random.choice(neg_index, size=n_neg, replace=False)
                    label[disable_index] = 0

            ret_label = torch.ones((anchor_boxes.size(0),), dtype=torch.long).mul_(-1).to(device)
            ret_label[index_inside] = label.long()
            ret_loc = torch.zeros((anchor_boxes.size(0), 4)).to(device)
            ret_loc[index_inside] = loc
            target_bboxes.append(ret_loc)
            target_labels.append(ret_label)

        return torch.stack(target_bboxes, 0), torch.stack(target_labels, 0)
