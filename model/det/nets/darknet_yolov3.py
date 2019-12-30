#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from model.det.layers.yolo_detection_layer import YOLODetectionLayer
from lib.model.module_helper import ModuleHelper


class DarkNetYolov3(nn.Module):
    def __init__(self, configer):
        super(DarkNetYolov3, self).__init__()
        self.configer = configer
        self.backbone = ModuleHelper.get_backbone(
            backbone=configer.get('network.backbone'),
            pretrained=configer.get('network.pretrained', default=None)
        )
        self.yolov3_head = Yolov3Head(configer, out_filters=self.backbone.num_features)
        self.yolo_detection_layer = YOLODetectionLayer(self.configer)

    def forward(self, data_dict):
        tuple_features = self.backbone(data_dict['img'])
        feat_list = self.yolov3_head(tuple_features)
        predictions, detections = self.yolo_detection_layer(feat_list)
        if 'testing' in data_dict and data_dict['testing']:
            return dict(dets=detections)

        loss = self.yolov3_loss(predictions, detections, feat_list, data_dict)
        return dict(dets=detections, pred=predictions, feat_list=feat_list, loss=loss)


class Yolov3Head(nn.Module):
    def __init__(self, configer, out_filters=None):
        super(Yolov3Head, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        #  backbone
        _out_filters = out_filters

        #  embedding0
        final_out_filter0 = len(self.configer.get("anchor", "anchors_list")[0]) * (5 + self.num_classes)

        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1])
        self.conv_out1 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)),
                ("bn", nn.BatchNorm2d(1024)),
                ("relu", nn.LeakyReLU(0.1)),
                ("conv_out", nn.Conv2d(1024, final_out_filter0, kernel_size=1, stride=1, padding=0, bias=True))
            ])
        )

        #  embedding1
        final_out_filter1 = len(self.configer.get("anchor", "anchors_list")[1]) * (5 + self.num_classes)

        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256)
        self.conv_out2 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
                ("bn", nn.BatchNorm2d(512)),
                ("relu", nn.LeakyReLU(0.1)),
                ("conv_out", nn.Conv2d(512, final_out_filter1, kernel_size=1, stride=1, padding=0, bias=True))
            ])
        )
        #  embedding2
        final_out_filter2 = len(self.configer.get("anchor", "anchors_list")[2]) * (5 + self.num_classes)

        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128)
        self.conv_out3 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
                ("bn", nn.BatchNorm2d(256)),
                ("relu", nn.LeakyReLU(0.1)),
                ("conv_out", nn.Conv2d(256, final_out_filter2, kernel_size=1, stride=1, padding=0, bias=True))
            ])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1)])
        return m

    def forward(self, tuple_features):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
            return _in

        #  yolo branch 0
        x0_in = _branch(self.embedding0, tuple_features[-1])
        out0 = self.conv_out1(x0_in)

        #  yolo branch 1
        x1_in = self.embedding1_cbl(x0_in)
        x1_in = torch.cat([F.interpolate(x1_in, scale_factor=2, mode='nearest'), tuple_features[-2]], 1)
        x1_in = _branch(self.embedding1, x1_in)
        out1 = self.conv_out2(x1_in)

        #  yolo branch 2
        x2_in = self.embedding2_cbl(x1_in)
        x2_in = torch.cat([F.interpolate(x2_in, scale_factor=2, mode='nearest'), tuple_features[-3]], 1)
        x2_in = _branch(self.embedding2, x2_in)
        out2 = self.conv_out3(x2_in)
        return [out0, out1, out2]
