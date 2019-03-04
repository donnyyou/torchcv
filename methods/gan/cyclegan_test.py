#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for GAN.


import torch

from methods.tools.blob_helper import BlobHelper
from methods.tools.runner_helper import RunnerHelper
from models.gan.model_manager import ModelManager
from datasets.test.data_loader import DataLoader
from utils.tools.logger import Logger as Log


class CycleGANTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.model_manager = ModelManager(configer)
        self.test_loader = DataLoader(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.gan_net = None

        self._init_model()

    def _init_model(self):
        self.gan_net = self.model_manager.gan_model()
        self.gan_net = RunnerHelper.load_net(self, self.gan_net)
        self.gan_net.eval()

    def test(self, imgA_dir=None, imgB_dir=None):
        if imgA_dir is not None:
            Log.info('ImageA Dir: {}'.format(imgA_dir))
            for data_dict in self.test_loader.get_testloader(imgA_dir):
                new_data_dict = dict(imgA=data_dict['img'])
                out_dict = self.gan_net(new_data_dict)

        if imgB_dir is not None:
            Log.info('ImageB Dir: {}'.format(imgB_dir))
            for data_dict in self.test_loader.get_testloader(imgB_dir):
                new_data_dict = dict(imgB=data_dict['img'])
                out_dict = self.gan_net(new_data_dict)




