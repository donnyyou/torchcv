#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import math
import numpy as np
import torch

from lib.tools.util.logger import Logger as Log


class SSDPriorBoxLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer, clip=False):
        self.configer = configer
        self.clip = clip

    def __call__(self, feat_list, input_size):
        img_w, img_h = input_size
        feature_map_w = [feat.size(3) for feat in feat_list]
        feature_map_h = [feat.size(2) for feat in feat_list]
        stride_w_list = [img_w / feat_w for feat_w in feature_map_w]
        stride_h_list = [img_h / feat_h for feat_h in feature_map_h]

        anchor_boxes_list = list()
        for i in range(len(feat_list)):
            fm_w = feature_map_w[i]
            fm_h = feature_map_h[i]
            boxes = []
            stride_offset_w, stride_offset_h = 0.5 * stride_w_list[i], 0.5 * stride_h_list[i]
            if self.configer.get('anchor', 'anchor_method') == 'ssd':
                s_w = self.configer.get('anchor', 'cur_anchor_sizes')[i]
                s_h = self.configer.get('anchor', 'cur_anchor_sizes')[i]
                boxes.append((stride_offset_w, stride_offset_h, s_w, s_h))
                extra_s = math.sqrt(self.configer.get('anchor', 'cur_anchor_sizes')[i]
                                    * self.configer.get('anchor', 'cur_anchor_sizes')[i + 1])

                boxes.append((stride_offset_w, stride_offset_h, extra_s, extra_s))

                for ar in self.configer.get('anchor', 'aspect_ratio_list')[i]:
                    boxes.append((stride_offset_w, stride_offset_h, s_w * math.sqrt(ar), s_h / math.sqrt(ar)))
                    boxes.append((stride_offset_w, stride_offset_h, s_w / math.sqrt(ar), s_h * math.sqrt(ar)))

            elif self.configer.get('anchor', 'anchor_method') == 'retina':
                s_w = self.configer.get('anchor', 'cur_anchor_sizes')[i]
                s_h = self.configer.get('anchor', 'cur_anchor_sizes')[i]
                for sr in self.configer.get('anchor', 'scale_ratio_list'):
                    s_w = sr * s_w
                    s_h = sr * s_h
                    for ar in self.configer.get('anchor', 'aspect_ratio_list'):
                        boxes.append((stride_offset_w, stride_offset_h, s_w * ar, s_h / ar))

            elif self.configer.get('anchor', 'anchor_method') == 'naive':
                s_w = self.configer.get('anchor', 'anchor_sizes_list')[i][0]
                s_h = self.configer.get('anchor', 'anchor_sizes_list')[i][1]
                for ar in self.configer.get('anchor', 'aspect_ratio_list')[i]:
                    boxes.append((stride_offset_w, stride_offset_h, s_w * math.sqrt(ar), s_h / math.sqrt(ar)))

            else:
                Log.error('Anchor Method {} not valid.'.format(self.configer.get('anchor', 'anchor_method')))
                exit(1)

            anchor_bases = torch.FloatTensor(np.array(boxes))
            assert anchor_bases.size(0) == self.configer.get('anchor', 'num_anchor_list')[i]
            anchors = anchor_bases.contiguous().view(1, -1, 4).repeat(fm_h * fm_w, 1, 1).contiguous().view(-1, 4)
            grid_len_h = np.arange(0, img_h - stride_offset_h, stride_h_list[i])
            grid_len_w = np.arange(0, img_w - stride_offset_w, stride_w_list[i])
            a, b = np.meshgrid(grid_len_w, grid_len_h)

            x_offset = torch.FloatTensor(a).view(-1, 1)
            y_offset = torch.FloatTensor(b).view(-1, 1)

            x_y_offset = torch.cat((x_offset, y_offset), 1).contiguous().view(-1, 1, 2)
            x_y_offset = x_y_offset.repeat(1, self.configer.get('anchor', 'num_anchor_list')[i], 1).contiguous().view(-1, 2)
            anchors[:, :2] = anchors[:, :2] + x_y_offset
            anchor_boxes_list.append(anchors)

        anchor_boxes = torch.cat(anchor_boxes_list, 0)
        if self.clip:
            anchor_boxes[:, 0::2].clamp_(min=0., max=img_w - 1)
            anchor_boxes[:, 1::2].clamp_(min=0., max=img_h - 1)

        return anchor_boxes

