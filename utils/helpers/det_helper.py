#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

import extensions.layers.nms.src.cython_nms as cython_nms
import extensions.layers.iou.src.cython_iou as cython_iou
from utils.tools.logger import Logger as Log


class DetHelper(object):

    @staticmethod
    def cython_nms(bboxes, scores=None, nms_threshold=0.3):
        """Apply classic DPM-style greedy NMS."""
        is_torch = False
        if not isinstance(bboxes, np.ndarray):
            bboxes = bboxes.cpu().numpy()
            if scores is None:
                scores = len(bboxes) - np.arange(len(bboxes), dtype=np.float32)
            else:
                scores = scores.cpu().numpy()

            is_torch = True

        if scores is None:
            scores = len(bboxes) - np.arange(len(bboxes), dtype=np.float32)

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1, 1)

        dets = np.concatenate((bboxes, scores), 1)
        if dets.shape[0] == 0:
            return []

        keep = cython_nms.nms(dets, nms_threshold)
        return keep if not is_torch else torch.from_numpy(keep).long()

    @staticmethod
    def cython_soft_nms(bboxes, scores=None, nms_threshold=0.0, score_threshold=0.001, sigma=0.5, method='linear'):
        """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
        is_torch = False
        if not isinstance(bboxes, np.ndarray):
            bboxes = bboxes.cpu().numpy()
            if scores is None:
                scores = len(bboxes) - np.arange(len(bboxes), dtype=np.float32)
            else:
                scores = scores.cpu().numpy()

            is_torch = True

        if scores is None:
            scores = len(bboxes) - np.arange(len(bboxes), dtype=np.float32)

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1, 1)

        dets = np.concatenate((bboxes, scores), 1)

        if dets.shape[0] == 0:
            return dets, []

        methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
        assert method in methods, 'Unknown soft_nms method: {}'.format(method)

        dets, keep = cython_nms.soft_nms(
            np.ascontiguousarray(dets, dtype=np.float32),
            np.float32(sigma),
            np.float32(nms_threshold),
            np.float32(score_threshold),
            np.uint8(methods[method])
        )
        return keep if not is_torch else torch.from_numpy(keep).long()

    @staticmethod
    def nms(bboxes, scores=None, nms_threshold=0.0, mode='union'):
        """Non maximum suppression.

        Args:
          bboxes(tensor): bounding boxes, sized [N,4].
          scores(tensor): bbox scores, sized [N,].
          threshold(float): overlap threshold.
          mode(str): 'union' or 'min'.

        Returns:
          keep(tensor): selected indices.

        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        """
        bboxes = bboxes.contiguous().view(-1, 4)
        if scores is not None:
            scores = scores.contiguous().view(-1,)

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        if scores is not None:
            _, order = scores.sort(0, descending=True)
        else:
            order = np.arange(len(bboxes), dtype=np.int32)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0]
            keep.append(i)
            xx1 = x1[order[1:]].clamp(min=x1[i].item())
            yy1 = y1[order[1:]].clamp(min=y1[i].item())
            xx2 = x2[order[1:]].clamp(max=x2[i].item())
            yy2 = y2[order[1:]].clamp(max=y2[i].item())

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= nms_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break

            order = order[ids + 1]

        return torch.LongTensor(keep)

    @staticmethod
    def cls_nms(bboxes, scores=None, labels=None, nms_threshold=0.0,
                iou_mode='union', cls_keep_num=None, nms_mode='nms',
                score_threshold=0.001, soft_sigma=0.5, soft_method='linear'):
        unique_labels = labels.cpu().unique()
        bboxes = bboxes.contiguous().view(-1, 4)
        if scores is not None:
            scores = scores.contiguous().view(-1,)

        if labels is not None:
            labels = labels.contiguous().view(-1,)

        unique_labels = unique_labels.to(bboxes.device)

        cls_keep_list = list()
        for c in unique_labels:
            cls_index = torch.nonzero(labels == c).squeeze(1)
            if nms_mode == 'nms':
                cls_keep = DetHelper.nms(bboxes[cls_index],
                                         scores=None if scores is None else scores[cls_index],
                                         nms_threshold=nms_threshold,
                                         mode=iou_mode)

            elif nms_mode == 'cython_nms':
                cls_keep = DetHelper.cython_nms(bboxes[cls_index],
                                                scores=None if scores is None else scores[cls_index],
                                                nms_threshold=nms_threshold)

            elif nms_mode == 'cython_soft_nms':
                cls_keep = DetHelper.cython_soft_nms(bboxes[cls_index],
                                                    scores=None if scores is None else scores[cls_index],
                                                    nms_threshold=nms_threshold,
                                                    score_threshold=score_threshold,
                                                    sigma=soft_sigma,
                                                    method=soft_method)

            else:
                Log.error('Not supported NMS mode: {}.'.format(nms_mode))
                exit(1)

            if cls_keep_num is not None:
                cls_keep = cls_keep[:cls_keep_num]

            cls_keep_list.append(cls_index[cls_keep])

        return torch.cat(cls_keep_list, 0)

    @staticmethod
    def cython_bbox_iou(bbox1, bbox2):
        is_torch = False
        if not isinstance(bbox1, np.ndarray):
            is_torch = True
            bbox1 = bbox1.numpy()
            bbox2 = bbox2.numpy()

        iou_matrix = cython_iou.bbox_overlaps(bbox1, bbox2)
        return iou_matrix if not is_torch else torch.from_numpy(iou_matrix)

    @staticmethod
    def bbox_iou(box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

        Args:
          box1(tensor): bounding boxes, sized [N,4]; [[xmin, ymin, xmax, ymax], ...]
          box2(tensor): bounding boxes, sized [M,4].
        Return:
          iou(tensor): sized [N,M].

        """
        if len(box1.size()) == 1:
            box1 = box1.unsqueeze(0)

        if len(box2.size()) == 1:
            box2 = box2.unsqueeze(0)

        N = box1.size(0)
        M = box2.size(0)

        # max(xmin, ymin).
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
        )

        # min(xmax, ymax)
        rb = torch.min(
            box1[:, 2:4].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:4].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    @staticmethod
    def bbox_kmeans(bboxes, cluster_number, dist=np.median):
        box_number = bboxes.shape[0]
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = bboxes[np.random.choice(box_number, cluster_number, replace=False)]  # init k clusters

        while True:
            distances = 1 - DetHelper.bbox_iou(torch.from_numpy(bboxes), torch.from_numpy(clusters))
            distances = distances.numpy()
            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change

            for cluster in range(cluster_number):
                clusters[cluster] = dist(  # update clusters
                    bboxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        result = clusters[np.lexsort(clusters.T[0, None])]
        avg_iou = DetHelper.avg_iou(bboxes, result)
        return result, avg_iou

    @staticmethod
    def avg_iou(boxes, clusters):
        iou_matrix = DetHelper.bbox_iou(torch.from_numpy(boxes), torch.from_numpy(clusters)).numpy()
        accuracy = np.mean([np.max(iou_matrix, axis=1)])
        return accuracy


if __name__ == "__main__":
    cluster_number = 9
    filename = "2012_train.txt"
    kmeans = DetHelper.bbox_kmeans(None, None)

