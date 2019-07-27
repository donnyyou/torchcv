#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Det Model for object detection.


import torch

from model.det.nets.darknet_yolov3 import DarkNetYolov3
from model.det.nets.vgg16_ssd300 import Vgg16SSD300
from model.det.nets.vgg16_ssd512 import Vgg16SSD512
from model.det.nets.faster_rcnn import FasterRCNN
from model.det.loss.det_modules import SSDMultiBoxLoss, SSDFocalLoss, YOLOv3Loss, FasterRCNNLoss
from util.tools.logger import Logger as Log

DET_MODEL_DICT = {
    'vgg16_ssd300': Vgg16SSD300,
    'vgg16_ssd512': Vgg16SSD512,
    'darknet_yolov3': DarkNetYolov3,
    'faster_rcnn': FasterRCNN,
}

DET_LOSS_DICT = {
    'ssd_multibox_loss': SSDMultiBoxLoss,
    'ssd_focal_loss': SSDFocalLoss,
    'yolov3_loss': YOLOv3Loss,
    'fasterrcnn_loss': FasterRCNNLoss
}


class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def object_detector(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in DET_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = DET_MODEL_DICT[model_name](self.configer)

        return model

    def get_det_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in DET_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = DET_LOSS_DICT[key](self.configer)
        if self.configer.get('network', 'loss_balance') and len(range(torch.cuda.device_count())) > 1:
            from exts.tools.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss
