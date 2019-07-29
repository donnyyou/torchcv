#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn
import torch.nn.functional as F

from model.det.layers.ssd_priorbox_layer import SSDPriorBoxLayer


class SSDDetectionLayer(nn.Module):

    def __init__(self, configer):
        super(SSDDetectionLayer, self).__init__()
        self.ssd_priorbox_layer = SSDPriorBoxLayer(configer)
        self.num_classes = configer.get('data', 'num_classes')
        self.num_anchors = configer.get('anchor', 'num_anchor_list')
        self.num_features = configer.get('network', 'num_feature_list')
        self.ssd_head_index = configer.get('network', 'head_index_list')

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        for i in range(max(self.ssd_head_index) + 1):
            self.loc_layers.append(
                nn.Conv2d(self.num_features[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)
            )
            self.conf_layers.append(
                nn.Conv2d(self.num_features[i], self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat_list, data_dict):
        y_locs = []
        y_confs = []

        for i, x in enumerate(feat_list):
            y_loc = self.loc_layers[self.ssd_head_index[i]](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[self.ssd_head_index[i]](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)
            y_confs.append(y_conf)

        pred_loc = torch.cat(y_locs, 1)
        pred_conf = torch.cat(y_confs, 1)
        input_size = [data_dict['img'].size(3), data_dict['img'].size(2)]
        default_boxes = self.ssd_priorbox_layer(feat_list, input_size)
        default_boxes = default_boxes.unsqueeze(0).repeat(pred_loc.size(0), 1, 1).to(pred_loc.device)

        variances = [0.1, 0.2]
        wh = torch.exp(pred_loc[:, :, 2:] * variances[1]) * default_boxes[:, :, 2:]
        cxcy = pred_loc[:, :, :2] * variances[0] * default_boxes[:, :, 2:] + default_boxes[:, :, :2]
        dets_loc = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 2)  # [b, 8732,4]
        # clip bounding box
        dets_loc[:, :, 0::2] = dets_loc[:, :, 0::2].clamp(min=0, max=input_size[0] - 1).div(input_size[0])
        dets_loc[:, :, 1::2] = dets_loc[:, :, 1::2].clamp(min=0, max=input_size[1] - 1).div(input_size[1])
        dets_conf = F.softmax(pred_conf, dim=-1)
        return pred_loc, pred_conf, dets_loc, dets_conf
