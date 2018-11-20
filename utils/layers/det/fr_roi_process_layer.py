#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

from utils.tools.logger import Logger as Log


class FRRoiProcessLayer(nn.Module):
    def __init__(self, configer):
        super(FRRoiProcessLayer, self).__init__()
        self.configer = configer
        if self.configer.get('roi', 'method') == 'roipool':
            from extensions.layers.roipool.module import RoIPool2D
            self.roi_layer = RoIPool2D(pooled_height=int(self.configer.get('roi', 'pooled_height')),
                                       pooled_width=int(self.configer.get('roi', 'pooled_width')),
                                       spatial_scale=1.0 / float(self.configer.get('roi', 'spatial_stride')))

        elif self.configer.get('roi', 'method') == 'roialign':
            from extensions.layers.roialign.module import RoIAlign2D
            self.roi_layer = RoIAlign2D(pooled_height=int(self.configer.get('roi', 'pooled_height')),
                                        pooled_width=int(self.configer.get('roi', 'pooled_width')),
                                        spatial_scale=1.0 / float(self.configer.get('roi', 'spatial_stride')),
                                        sampling_ratio=2)

        else:
            Log.error('Invalid roi method.')
            exit(1)

    def forward(self, features, rois, scale=None):
        out = self.roi_layer(features, rois, scale)
        return out

