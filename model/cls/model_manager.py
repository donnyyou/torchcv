#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Cls Model for semantic segmentation.


from model.cls.nets.base_model import BaseModel
from model.cls.nets.cls_model import ClsModel
from model.cls.nets.distill_model import DistillModel
from model.cls.loss.loss import Loss
from lib.tools.util.logger import Logger as Log


CLS_MODEL_DICT = {
    'base_model': BaseModel,
    'cls_model': ClsModel,
    'distill_model': DistillModel,
}


class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def get_cls_model(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in CLS_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = CLS_MODEL_DICT[model_name](self.configer)
        return model

    def get_cls_loss(self):
        if self.configer.get('network', 'gather'):
            return Loss(self.configer)

        from lib.parallel.data_parallel import ParallelCriterion
        return ParallelCriterion(Loss(self.configer))
