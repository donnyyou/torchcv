#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Single Shot Detector data loader


import os
import json
import random
import torch.utils.data as data

from lib.parallel.data_container import DataContainer
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as Log


class DefaultFaceGANDataset(data.Dataset):

    def __init__(self, root_dir=None, dataset=None, tag=None,
                 aug_transform=None, img_transform=None, configer=None):
        super(DefaultFaceGANDataset, self).__init__()
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.imgA_list, self.labelA_list, self.imgB_list, self.labelB_list = self.__read_json_file(root_dir, dataset, tag)

    def __getitem__(self, index):
        imgA = ImageHelper.read_image(self.imgA_list[index],
                                      tool=self.configer.get('data', 'image_tool'),
                                      mode=self.configer.get('data', 'input_mode'))

        indexB = random.randint(0, len(self.imgB_list) - 1) % len(self.imgB_list)
        imgB = ImageHelper.read_image(self.imgB_list[indexB],
                                      tool=self.configer.get('data', 'image_tool'),
                                      mode=self.configer.get('data', 'input_mode'))

        if self.aug_transform is not None:
            imgA = self.aug_transform(imgA)
            imgB = self.aug_transform(imgB)

        if self.img_transform is not None:
            imgA = self.img_transform(imgA)
            imgB = self.img_transform(imgB)

        return dict(
            imgA=DataContainer(imgA, stack=True),
            imgB=DataContainer(imgB, stack=True),
            labelA=DataContainer(self.labelA_list[index], stack=True),
            labelB=DataContainer(self.labelB_list[indexB], stack=True)
        )

    def __len__(self):

        return len(self.imgA_list)

    def __read_json_file(self, root_dir, dataset, tag):
        imgA_list = list()
        imgB_list = list()
        labelA_list = list()
        labelB_list = list()

        with open(os.path.join(root_dir, 'protocols/{}_label{}A.json'.format(dataset, tag)), 'r') as file_stream:
            items = json.load(file_stream)
            for item in items:
                img_path = os.path.join(root_dir, item['image_path'])
                if not os.path.exists(img_path):
                    Log.warn('Image Path: {} not exists.'.format(img_path))
                    continue

                imgA_list.append(img_path)
                labelA_list.append(item['label'])

        with open(os.path.join(root_dir, 'protocols/{}_label{}B.json'.format(dataset, tag)), 'r') as file_stream:
            items = json.load(file_stream)
            for item in items:
                img_path = os.path.join(root_dir, item['image_path'])
                if not os.path.exists(img_path):
                    Log.warn('Image Path: {} not exists.'.format(img_path))
                    continue

                imgB_list.append(img_path)
                labelB_list.append(item['label'])

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            with open(os.path.join(root_dir, 'protocols/val_label{}A.json'.format(tag)), 'r') as file_stream:
                items = json.load(file_stream)
                for item in items:
                    img_path = os.path.join(root_dir, item['image_path'])
                    if not os.path.exists(img_path):
                        Log.warn('Image Path: {} not exists.'.format(img_path))
                        continue

                    imgA_list.append(img_path)
                    labelA_list.append(item['label'])

            with open(os.path.join(root_dir, 'protocols/val_label{}B.json'.format(tag)), 'r') as file_stream:
                items = json.load(file_stream)
                for item in items:
                    img_path = os.path.join(root_dir, item['image_path'])
                    if not os.path.exists(img_path):
                        Log.warn('Image Path: {} not exists.'.format(img_path))
                        continue

                    imgB_list.append(img_path)
                    labelB_list.append(item['label'])

        return imgA_list, labelA_list, imgB_list, labelB_list
