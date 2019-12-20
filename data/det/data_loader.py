#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
from torch.utils import data

import lib.data.pil_aug_transforms as pil_aug_trans
import lib.data.cv2_aug_transforms as cv2_aug_trans
import lib.data.transforms as trans
from lib.data.collate import collate
from lib.tools.util.logger import Logger as Log
from data.det.datasets.default_dataset import DefaultDataset


class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_train_transform = pil_aug_trans.PILAugCompose(self.configer, split='train')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='train')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_val_transform = pil_aug_trans.PILAugCompose(self.configer, split='val')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_val_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='val')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(**self.configer.get('data', 'normalize')), ])

    def get_trainloader(self):
        if self.configer.get('dataset', default=None) in [None, 'default']:
            dataset = DefaultDataset(root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                                     aug_transform=self.aug_train_transform,
                                     img_transform=self.img_transform, configer=self.configer)

        else:
            Log.error('{} dataset is invalid.'.format(self.configer.get('dataset')))
            exit(1)

        sampler = None
        if self.configer.get('network.distributed'):
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        trainloader = data.DataLoader(
            dataset, sampler=sampler,
            batch_size=self.configer.get('train', 'batch_size'), shuffle=(sampler is None),
            num_workers=self.configer.get('data', 'workers'), pin_memory=True,
            drop_last=self.configer.get('data', 'drop_last'),
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('train', 'data_transformer')
            )
        )

        return trainloader

    def get_valloader(self):
        if self.configer.get('dataset', default=None) in [None, 'default']:
            dataset = DefaultDataset(root_dir=self.configer.get('data', 'data_dir'), dataset='val',
                                     aug_transform=self.aug_val_transform,
                                     img_transform=self.img_transform,
                                     configer=self.configer)

        else:
            Log.error('{} dataset is invalid.'.format(self.configer.get('dataset')))
            exit(1)

        valloader = data.DataLoader(
            dataset,
            batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
            num_workers=self.configer.get('data', 'workers'), pin_memory=True,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('val', 'data_transformer')
            )
        )

        return valloader

