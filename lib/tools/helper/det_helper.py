#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import numpy as np
import torch

from lib.tools.util.logger import Logger as Log
try:
    from lib.exts.ops.nms.nms_wrapper import nms
except ImportError:
    print('DetHelper NMS ImportError.')

try:
    from lib.exts.ops.nms.nms_wrapper import soft_nms
except ImportError:
    print('DetHelper Soft-NMS ImportError.')


class DetHelper(object):

    @staticmethod
    def cls_nms(dets, labels, max_threshold=0.0, cls_keep_num=None, device_id=None, return_ind=False):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        assert isinstance(labels, np.ndarray)

        unique_labels = np.unique(labels)

        cls_keep_list = list()
        for c in unique_labels:
            cls_index = np.where(labels == c)[0]
            _, cls_keep = nms(dets[cls_index], iou_thr=max_threshold, device_id=device_id)

            if cls_keep_num is not None:
                cls_keep = cls_keep[:cls_keep_num]

            cls_keep_list.append(cls_index[cls_keep.cpu().numpy()])

        keep_index = np.concatenate(cls_keep_list, 0)
        return keep_index if return_ind else dets[keep_index]

    @staticmethod
    def cls_softnms(dets, labels, max_threshold=0.0, min_score=0.001, sigma=0.5, method='linear', cls_keep_num=None):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        assert isinstance(labels, np.ndarray)

        unique_labels = np.unique(labels)

        cls_dets_list = list()
        for c in unique_labels:
            cls_index = np.where(labels == c)[0]
            cls_dets, _ = soft_nms(dets[cls_index], iou_thr=max_threshold,
                                method=method, sigma=sigma, min_score=min_score)

            if cls_keep_num is not None:
                cls_dets = cls_dets[:cls_keep_num]

            cls_dets_list.append(cls_dets)

        if isinstance(cls_dets_list[0], torch.Tensor):
            return torch.cat(cls_dets_list, 0)

        return np.concatenate(cls_dets_list, 0)

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
    def bbox_kmeans(bboxes, cluster_number, dist=np.mean):
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

