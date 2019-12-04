#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import math
import numpy as np
import torch


class PafGenerator(object):

    def __init__(self, configer):
        self.configer = configer

    def __call__(self, gt_kpts, input_size, maskmap=None):
        input_width, input_height = input_size
        vec_pair = self.configer.get('details', 'limb_seq')
        stride = self.configer.get('network', 'stride')
        theta = self.configer.get('target.paf.theta')
        width, height = input_width // stride, input_height // stride
        accumulate_vec_map = np.zeros((len(vec_pair) * 2, height, width), dtype=np.float32)
        cnt = np.zeros((len(vec_pair), height, width), dtype=np.int32)
        channel, height, width = cnt.shape
        for j in range(len(gt_kpts)):
            for i in range(channel):
                a = vec_pair[i][0] - 1
                b = vec_pair[i][1] - 1
                if gt_kpts[j][a][2] < 0 or gt_kpts[j][b][2] < 0:
                    continue

                ax = gt_kpts[j][a][0].item() / stride
                ay = gt_kpts[j][a][1].item() / stride
                bx = gt_kpts[j][b][0].item() / stride
                by = gt_kpts[j][b][1].item() / stride

                bax = bx - ax
                bay = by - ay
                # 1e-9 to aviod two points have same position.
                norm_ba = math.sqrt(bax * bax + bay * bay)
                if norm_ba == 0:
                    continue

                bax /= norm_ba
                bay /= norm_ba

                min_w = max(int(round(min(ax, bx) - theta)), 0)
                max_w = min(int(round(max(ax, bx) + theta)), width)
                min_h = max(int(round(min(ay, by) - theta)), 0)
                max_h = min(int(round(max(ay, by) + theta)), height)

                range_x = list(range(int(min_w), int(max_w), 1))
                range_y = list(range(int(min_h), int(max_h), 1))
                xx, yy = np.meshgrid(range_x, range_y)

                xx = xx.astype(np.uint32)
                yy = yy.astype(np.uint32)
                px = xx - ax  # the vector from (x,y) to centerA
                py = yy - ay
                limb_width = np.abs(bax * py - bay * px)
                mask = limb_width < theta  # mask is 2D

                vec_map = np.zeros((2, height, width), dtype=np.float32)
                vec_map[:, yy, xx] = np.repeat(mask[np.newaxis, :, :], 2, axis=0)
                vec_map[:, yy, xx] *= np.array([bax, bay])[:, np.newaxis, np.newaxis]

                mask = np.logical_or(np.abs(vec_map[0:1, :, :]) > 0, np.abs(vec_map[1:2, :, :]) > 0)

                accumulate_vec_map[2*i:2*i+2, :, :] = np.multiply(accumulate_vec_map[2*i:2*i+2], cnt[i:i+1, :, :])
                accumulate_vec_map[2*i:2*i+2, :, :] += vec_map
                cnt[i:i+1, :, :][mask == 1] += 1
                mask = cnt[i:i+1, :, :] == 0
                cnt[i:i+1, :, :][mask == 1] = 1
                accumulate_vec_map[2*i:2*i+2, :, :] = np.divide(accumulate_vec_map[2*i:2*i+2, :, :], cnt[i:i+1, :, :])
                cnt[i:i+1, :, :][mask == 1] = 0

        vecmap = torch.from_numpy(accumulate_vec_map)
        if maskmap is not None:
            vecmap = vecmap * maskmap

        return vecmap
