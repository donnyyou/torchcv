#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Single Shot Detector data loader


import os
import torch.utils.data as data

from extensions.tools.parallel import DataContainer
from utils.helpers.json_helper import JsonHelper
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class JsonLoader(data.Dataset):

    def __init__(self, root_dir=None, json_path=None, aug_transform=None, img_transform=None, configer=None):
        super(JsonLoader, self).__init__()
        self.configer = configer
        self.aug_transform=aug_transform
        self.img_transform = img_transform
        self.img_list = self.__read_json(root_dir, json_path)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))

        ori_img_size = ImageHelper.get_size(img)
        if self.aug_transform is not None:
            img = self.aug_transform(img)

        border_hw = ImageHelper.get_size(img)[::-1]
        if self.img_transform is not None:
            img = self.img_transform(img)

        meta = dict(
            ori_img_size=ori_img_size,
            border_hw=border_hw,
            img_path=self.img_list[index]
        )
        return dict(
            img=DataContainer(img, stack=True, return_dc=True, samples_per_gpu=True),
            meta=DataContainer(meta, stack=False, cpu_only=True, return_dc=True, samples_per_gpu=True)
        )

    def __len__(self):

        return len(self.img_list)

    def __read_json(self, root_dir, json_path):
        img_list = []
        for item in JsonHelper.load_file(json_path):
            img_path = os.path.join(root_dir, item['image_path'])
            if not os.path.exists(img_path) or not ImageHelper.is_img(img_path):
                Log.error('Image Path: {} is Invalid.'.format(img_path))
                exit(1)

            img_list.append(img_path)

        Log.info('There are {} images..'.format(len(img_list)))
        return img_list
