#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Det Model for object detection.


from model.det.nets.darknet_yolov3 import DarkNetYolov3
from model.det.nets.vgg16_ssd300 import Vgg16SSD300
from model.det.nets.vgg16_ssd512 import Vgg16SSD512
from model.det.nets.lffdv2 import LFFDv2
from model.det.nets.faster_rcnn import FasterRCNN
from model.det.loss.loss import Loss
from lib.tools.util.logger import Logger as Log

DET_MODEL_DICT = {
    'vgg16_ssd300': Vgg16SSD300,
    'vgg16_ssd512': Vgg16SSD512,
    'lffdv2': LFFDv2,
    'darknet_yolov3': DarkNetYolov3,
    'faster_rcnn': FasterRCNN,
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

    def get_det_loss(self):
        if self.configer.get('network', 'gather'):
            return Loss(self.configer)

        from lib.parallel.data_parallel import ParallelCriterion
        return ParallelCriterion(Loss(self.configer))
