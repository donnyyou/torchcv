#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
from torch.utils import data

import lib.data.pil_aug_transforms as pil_aug_trans
import lib.data.cv2_aug_transforms as cv2_aug_trans
from lib.data.collate import collate
from lib.data.transforms import ToTensor, Normalize, Compose
from lib.tools.util.logger import Logger as Log
from data.test.datasets.default_dataset import DefaultDataset
from data.test.datasets.facegan_dataset import FaceGANDataset
from data.test.datasets.list_dataset import ListDataset
from data.test.datasets.json_dataset import JsonDataset


class TestDataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_test_transform = pil_aug_trans.PILAugCompose(self.configer, split='test')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_test_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='test')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = Compose([
            ToTensor(),
            Normalize(**self.configer.get('data', 'normalize')), ])

    def get_testloader(self, test_dir=None, list_path=None, json_path=None):
        if self.configer.get('test.dataset', default=None) in [None, 'default']:
            test_dir = test_dir if test_dir is not None else self.configer.get('test', 'test_dir')
            dataset = DefaultDataset(test_dir=test_dir,
                                     aug_transform=self.aug_test_transform,
                                     img_transform=self.img_transform,
                                     configer=self.configer)

        elif self.configer.get('test.dataset') == 'list':
            list_path = list_path if list_path is not None else self.configer.get('test', 'list_path')
            dataset = ListDataset(root_dir=self.configer.get('test', 'root_dir'),
                                  list_path=list_path,
                                  aug_transform=self.aug_test_transform,
                                  img_transform=self.img_transform,
                                  configer=self.configer)

        elif self.configer.get('test.dataset') == 'json':
            json_path = json_path if json_path is not None else self.configer.get('test', 'json_path')
            dataset = JsonDataset(root_dir=self.configer.get('test', 'root_dir'),
                                  json_path=json_path,
                                  aug_transform=self.aug_test_transform,
                                  img_transform=self.img_transform,
                                  configer=self.configer)

        elif self.configer.get('test.dataset') == 'facegan':
            json_path = json_path if json_path is not None else self.configer.get('test', 'json_path')
            dataset = FaceGANDataset(root_dir=self.configer.get('test', 'root_dir'),
                                     json_path=json_path,
                                     aug_transform=self.aug_test_transform,
                                     img_transform=self.img_transform,
                                     configer=self.configer)

        else:
            Log.error('{} test dataset is invalid.'.format(self.configer.get('test.dataset')))
            exit(1)

        testloader = data.DataLoader(
            dataset,
            batch_size=self.configer.get('test.batch_size', default=torch.cuda.device_count()), shuffle=False,
            num_workers=self.configer.get('data', 'workers'), pin_memory=True,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('test', 'data_transformer')
            )
        )

        return testloader


