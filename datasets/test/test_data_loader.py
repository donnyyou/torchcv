#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from torch.utils import data

from datasets.test.loader.default_loader import DefaultLoader
from datasets.tools.collate import collate
from datasets.tools.transforms import ToTensor, Normalize, Compose
from utils.tools.logger import Logger as Log


class TestDataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        self.img_transform = Compose([
            ToTensor(),
            Normalize(**self.configer.get('data', 'normalize')), ])

    def get_testloader(self, test_dir=None, list_path=None):
        test_dir = test_dir if test_dir is not None else self.configer.get('test', 'data_dir')
        if not self.configer.exists('test', 'loader') or self.configer.get('test', 'loader') == 'default':
            trainloader = data.DataLoader(
                DefaultLoader(test_dir=test_dir,
                              list_path=list_path,
                              img_transform=self.img_transform,
                              configer=self.configer),
                batch_size=self.configer.get('test', 'batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True,
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('test', 'data_transformer')
                )
            )

            return trainloader

        else:
            Log.error('{} train loader is invalid.'.format(self.configer.get('train', 'loader')))
            exit(1)

