#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

import extensions.layers.nms.src.cython_nms as cython_nms


class NMS(object):
    def __init__(self):
        pass

    def __call__(self, bboxes, scores, max_threshold=0.3):
        """Apply classic DPM-style greedy NMS."""
        is_torch = False
        if not isinstance(bboxes, np.ndarray):
            bboxes = bboxes.numpy()
            scores = scores.numpy()
            is_torch = True

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1, 1)

        dets = np.concatenate((bboxes, scores), 1)
        if dets.shape[0] == 0:
            return []

        keep = cython_nms.nms(dets, max_threshold)
        return keep if not is_torch else torch.from_numpy(keep).long()


class SoftNMS(object):
    def __init__(self, sigma=0.5, method='linear'):
        self.sigma = sigma
        self.method = method

    def __call__(self, bboxes, scores, max_threshold=0.3, score_threshold=0.001):
        """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
        is_torch = False
        if not isinstance(bboxes, np.ndarray):
            bboxes = bboxes.numpy()
            scores = scores.numpy()
            is_torch = True

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1, 1)

        dets = np.concatenate((bboxes, scores), 1)

        if dets.shape[0] == 0:
            return dets, []

        methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
        assert self.method in methods, 'Unknown soft_nms method: {}'.format(self.method)

        dets, keep = cython_nms.soft_nms(
            np.ascontiguousarray(dets, dtype=np.float32),
            np.float32(self.sigma),
            np.float32(max_threshold),
            np.float32(score_threshold),
            np.uint8(methods[self.method])
        )
        return keep if not is_torch else torch.from_numpy(keep).long()