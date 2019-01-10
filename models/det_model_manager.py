#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Det Model for object detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.det.nets.darknet_yolov2 import DarkNetYolov2
from models.det.nets.darknet_yolov3 import DarkNetYolov3
from models.det.nets.vgg300_ssd import Vgg300SSD
from models.det.nets.vgg512_ssd import Vgg512SSD
from models.det.nets.faster_rcnn import FasterRCNN
from utils.tools.logger import Logger as Log

DET_MODEL_DICT = {
    'vgg300_ssd': Vgg300SSD,
    'vgg512_ssd': Vgg512SSD,
    'darknet_yolov2': DarkNetYolov2,
    'darknet_yolov3': DarkNetYolov3,
    'faster_rcnn': FasterRCNN,
}


class DetModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def object_detector(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in DET_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = DET_MODEL_DICT[model_name](self.configer)

        return model
