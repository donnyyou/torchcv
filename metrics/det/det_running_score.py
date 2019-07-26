#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Object Detection running score.


import time
import numpy as np


class DetRunningScore(object):
    def __init__(self, configer):
        self.configer = configer
        self.gt_list = list()
        self.pred_list = list()
        self.num_positive = list()

        for i in range(self.configer.get('data', 'num_classes')):
            self.gt_list.append(dict())
            self.pred_list.append(list())
            self.num_positive.append(1e-9)

    def _voc_ap(self, rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
            Compute VOC AP given precision and recall.
            If use_07_metric is true, uses the
            VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _voc_eval(self, iou_threshold=0.5, use_07_metric=False):

        ap_list = list()
        rc_list = list()
        pr_list = list()
        for i in range(self.configer.get('data', 'num_classes')):

            class_recs = self.gt_list[i]
            pred_recs = self.pred_list[i]
            for key in class_recs.keys():
                class_recs[key]['det'] = [False] * class_recs[key]['bbox'].shape[0]

            image_ids = np.array([pred_rec[0] for pred_rec in pred_recs])
            confidence = np.array([pred_rec[1] for pred_rec in pred_recs])
            BB = np.array([pred_rec[2] for pred_rec in pred_recs])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            BB = BB[sorted_ind]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                           (BBGT[:, 2] - BBGT[:, 0]) *
                           (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > iou_threshold:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(self.num_positive[i])
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec, use_07_metric=use_07_metric)
            rc_list.append(rec)
            ap_list.append(ap)
            pr_list.append(prec)

        return rc_list, pr_list, ap_list

    def update(self, batch_pred_bboxes, batch_gt_bboxes, batch_gt_labels):
        image_name_prefix = str(time.time())
        for i in range(len(batch_gt_bboxes)):
            image_name = '{}_{}'.format(image_name_prefix, i)
            for cls in range(self.configer.get('data', 'num_classes')):
                self.gt_list[cls][image_name] = {
                    'bbox': np.array([batch_gt_bboxes[i][j].cpu().numpy()
                                      for j in range(batch_gt_bboxes[i].size(0))
                                      if batch_gt_labels[i][j] == cls])
                }

                self.num_positive[cls] += (self.gt_list[cls][image_name]['bbox']).shape[0]

            for pred_box in batch_pred_bboxes[i]:
                self.pred_list[pred_box[4]].append([image_name, pred_box[5], pred_box[:4]])

    def get_mAP(self):
        # compute mAP by APs under different oks thresholds
        use_07_metric = self.configer.get('val', 'use_07_metric')
        rc_list, pr_list, ap_list = self._voc_eval(use_07_metric=use_07_metric)
        if self.num_positive[self.configer.get('data', 'num_classes') - 1] < 1:
            return sum(ap_list) / (self.configer.get('data', 'num_classes') - 1)
        else:
            return sum(ap_list) / self.configer.get('data', 'num_classes')

    def reset(self):
        self.gt_list = list()
        self.pred_list = list()
        self.num_positive = list()

        for i in range(self.configer.get('data', 'num_classes')):
            self.gt_list.append(dict())
            self.pred_list.append(list())
            self.num_positive.append(1e-9)
