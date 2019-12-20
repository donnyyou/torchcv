#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select GAN Model for object detection.


from model.gan.nets.pix2pix import Pix2Pix
from model.gan.nets.cycle_gan import CycleGAN
from lib.tools.util.logger import Logger as Log

GAN_MODEL_DICT = {
    'pix2pix': Pix2Pix,
    'cyclegan': CycleGAN,
}


class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def gan_model(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in GAN_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = GAN_MODEL_DICT[model_name](self.configer)

        return model
