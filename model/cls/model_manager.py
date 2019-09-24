#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Cls Model for semantic segmentation.


import model.backbone as models
from model.cls.nets.cls_model import ClsModel
from model.cls.nets.distill_model import DistillModel
from model.cls.loss.loss import Loss
from tools.util.logger import Logger as Log


CLS_MODEL_DICT = {
    'cls_model': ClsModel,
    'distill_model': DistillModel,
}


class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def get_cls_model(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in CLS_MODEL_DICT and model_name not in models.__dict__:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        if model_name in CLS_MODEL_DICT:
            model = CLS_MODEL_DICT[model_name](self.configer)

        else:
            model = models.__dict__[model_name](num_classes=self.configer.get('data.num_classes'))

        return model

    def get_cls_loss(self):
        if self.configer.get('network', 'gather'):
            return Loss(self.configer)

        from exts.tools.parallel.data_parallel import ParallelCriterion
        return ParallelCriterion(Loss(self.configer))
