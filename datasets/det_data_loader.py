#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch.utils import data

import datasets.tools.pil_aug_transforms as pil_aug_trans
import datasets.tools.cv2_aug_transforms as cv2_aug_trans
import datasets.tools.transforms as trans
from datasets.det.ssd_data_loader import SSDDataLoader
from datasets.det.fr_data_loader import FRDataLoader
from datasets.det.yolo_data_loader import YOLODataLoader
from datasets.tools.collate import collate
from utils.tools.logger import Logger as Log


class DetDataLoader(object):

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
            trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ])

    def get_trainloader(self):
        if self.configer.get('method') == 'single_shot_detector':
            trainloader = data.DataLoader(
                SSDDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                              aug_transform=self.aug_train_transform,
                              img_transform=self.img_transform,
                              configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True,
                drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('train', 'data_transformer')
                )
            )

            return trainloader

        elif self.configer.get('method') == 'faster_rcnn':
            trainloader = data.DataLoader(
                FRDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                             aug_transform=self.aug_train_transform,
                             img_transform=self.img_transform,
                             configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True,
                drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('train', 'data_transformer')
                )
            )

            return trainloader

        elif self.configer.get('method') == 'yolov3':
            trainloader = data.DataLoader(
                YOLODataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                               aug_transform=self.aug_train_transform,
                               img_transform=self.img_transform,
                               configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True,
                drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('train', 'data_transformer')
                )
            )

            return trainloader

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None

    def get_valloader(self):
        if self.configer.get('method') == 'single_shot_detector':
            valloader = data.DataLoader(
                SSDDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'val'),
                              aug_transform=self.aug_val_transform,
                              img_transform=self.img_transform,
                              configer=self.configer),
                batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True,
                drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('val', 'data_transformer')
                )
            )

            return valloader

        elif self.configer.get('method') == 'faster_rcnn':
            valloader = data.DataLoader(
                FRDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'val'),
                             aug_transform=self.aug_val_transform,
                             img_transform=self.img_transform,
                             configer=self.configer),
                batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True,
                drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('val', 'data_transformer')
                )
            )

            return valloader

        elif self.configer.get('method') == 'yolov3':
            valloader = data.DataLoader(
                YOLODataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'val'),
                               aug_transform=self.aug_val_transform,
                               img_transform=self.img_transform,
                               configer=self.configer),
                batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True,
                drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('val', 'data_transformer')
                )
            )

            return valloader

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None

