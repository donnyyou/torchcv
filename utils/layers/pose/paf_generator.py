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


class PafGenerator(object):

    def __init__(self, configer):
        self.configer = configer

    def __call__(self, gt_kpts, input_size, maskmap=None):
        width, height = input_size
        vec_pair = self.configer.get('details', 'limb_seq')
        stride = self.configer.get('network', 'stride')
        theta = self.configer.get('heatmap', 'theta')
        batch_size = len(gt_kpts)
        vecmap = np.zeros((batch_size, len(vec_pair) * 2, height // stride, width // stride), dtype=np.float32)
        start = stride / 2.0 - 0.5

        for batch_id in range(batch_size):
            cnt = np.zeros((len(vec_pair), height // stride, width // stride), dtype=np.int32)
            channel, height, width = cnt.shape
            for j in range(len(gt_kpts[batch_id])):
                for i in range(channel):
                    a = vec_pair[i][0] - 1
                    b = vec_pair[i][1] - 1
                    if gt_kpts[batch_id][j][a][2] < 0 or gt_kpts[batch_id][j][b][2] < 0:
                        continue

                    ax = (gt_kpts[batch_id][j][a][0].item() - start) * 1.0 / stride
                    ay = (gt_kpts[batch_id][j][a][1].item() - start) * 1.0 / stride
                    bx = (gt_kpts[batch_id][j][b][0].item() - start) * 1.0 / stride
                    by = (gt_kpts[batch_id][j][b][1].item() - start) * 1.0 / stride

                    bax = bx - ax
                    bay = by - ay
                    # 1e-9 to aviod two points have same position.
                    norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
                    bax /= norm_ba
                    bay /= norm_ba

                    min_w = max(int(round(min(ax, bx) - theta)), 0)
                    max_w = min(int(round(max(ax, bx) + theta)), width-1)
                    min_h = max(int(round(min(ay, by) - theta)), 0)
                    max_h = min(int(round(max(ay, by) + theta)), height-1)

                    for h in range(min_h, max_h+1):
                        for w in range(min_w, max_w+1):
                            px = w - ax
                            py = h - ay

                            dis = abs(bay * px - bax * py)
                            if dis <= theta:
                                temp = vecmap[batch_id][2*i][h][w]
                                vecmap[batch_id][2*i][h][w] = (temp * cnt[i][h][w] + bax) / (cnt[i][h][w] + 1)
                                temp = vecmap[batch_id][2*i+1][h][w]
                                vecmap[batch_id][2*i+1][h][w] = (temp * cnt[i][h][w] + bay) / (cnt[i][h][w] + 1)
                                cnt[i][h][w] += 1

        vecmap = torch.from_numpy(vecmap)
        if maskmap is not None:
            vecmap = vecmap * maskmap.unsqueeze(1)

        return vecmap
