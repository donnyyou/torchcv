#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn


class RPNDetectionLayer(nn.Module):
    def __init__(self, configer):
        super(RPNDetectionLayer, self).__init__()
        self.configer = configer
        self.num_anchors = self.configer.get('rpn', 'num_anchor_list')
        self.num_features = configer.get('rpn', 'num_feature_list')
        self.rpn_head_index = configer.get('rpn', 'head_index_list')

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        for i in range(max(self.rpn_head_index) + 1):
            self.loc_layers.append(
                nn.Conv2d(self.num_features[i], self.num_anchors[i] * 4, kernel_size=1, padding=0)
            )
            self.conf_layers.append(
                nn.Conv2d(self.num_features[i], self.num_anchors[i] * 2, kernel_size=1, padding=0)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat_list):
        y_locs = []
        y_confs = []

        for i, x in enumerate(feat_list):
            y_loc = self.loc_layers[self.rpn_head_index[i]](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[self.rpn_head_index[i]](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, 2)
            y_confs.append(y_conf)

        rpn_locs = torch.cat(y_locs, 1)
        rpn_scores = torch.cat(y_confs, 1)

        return rpn_locs, rpn_scores
