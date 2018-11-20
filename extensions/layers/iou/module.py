#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

import extensions.layers.iou.src.cython_iou as cython_iou


class BoxIOU(object):
    def __init__(self):
        pass

    def __call__(self, bbox1, bbox2):
        is_torch = False
        if not isinstance(bbox1, np.ndarray):
            is_torch = True
            bbox1 = bbox1.numpy()
            bbox2 = bbox2.numpy()

        iou_matrix = cython_iou.bbox_overlaps(bbox1, bbox2)
        return iou_matrix if not is_torch else torch.from_numpy(iou_matrix)