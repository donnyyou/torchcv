#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch

from utils.tools.logger import Logger as Log


class HeatmapGenerator(object):

    def __init__(self, configer):
        self.configer = configer

    def __call__(self, gt_kpts, input_size, maskmap=None):
        width, height = input_size
        stride = self.configer.get('network', 'stride')
        num_keypoints = self.configer.get('data', 'num_kpts')
        sigma = self.configer.get('heatmap', 'sigma')
        method = self.configer.get('heatmap', 'method')
        batch_size = len(gt_kpts)

        heatmap = np.zeros((batch_size, num_keypoints + 1, height // stride, width // stride), dtype=np.float32)
        start = stride / 2.0 - 0.5

        for batch_id in range(batch_size):
            for i in range(len(gt_kpts[batch_id])):
                for j in range(num_keypoints):
                    if gt_kpts[batch_id][i][j][2] < 0:
                        continue

                    x = gt_kpts[batch_id][i][j][0]
                    y = gt_kpts[batch_id][i][j][1]
                    for h in range(height // stride):
                        for w in range(width // stride):
                            xx = start + w * stride
                            yy = start + h * stride
                            dis = 0
                            if method == 'gaussian':
                                dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                            elif method == 'laplace':
                                dis = math.sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma
                            else:
                                Log.error('Method: {} is not valid.'.format(method))
                                exit(1)

                            if dis > 4.6052:
                                continue

                            # Use max operator?
                            heatmap[batch_id][j][h][w] = max(math.exp(-dis), heatmap[batch_id][j][h][w])
                            if heatmap[batch_id][j][h][w] > 1:
                                heatmap[batch_id][j][h][w] = 1

            heatmap[batch_id, num_keypoints, :, :] = 1.0 - np.max(heatmap[batch_id, :-1, :, :], axis=0)

        heatmap = torch.from_numpy(heatmap)
        if maskmap is not None:
            heatmap = heatmap * maskmap.unsqueeze(1)

        return heatmap
