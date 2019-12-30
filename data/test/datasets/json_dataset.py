#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Single Shot Detector data loader


import os
import torch.utils.data as data

from lib.parallel.data_container import DataContainer
from lib.tools.helper.json_helper import JsonHelper
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as Log


class JsonDataset(data.Dataset):

    def __init__(self, root_dir=None, json_path=None, aug_transform=None, img_transform=None, configer=None):
        super(JsonDataset, self).__init__()
        self.configer = configer
        self.aug_transform=aug_transform
        self.img_transform = img_transform
        self.item_list = self.__read_json(root_dir, json_path)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.item_list[index][0],
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
            img_path=self.item_list[index][0],
            filename=self.item_list[index][1]
        )
        return dict(
            img=DataContainer(img, stack=True, return_dc=True, samples_per_gpu=True),
            meta=DataContainer(meta, stack=False, cpu_only=True, return_dc=True, samples_per_gpu=True)
        )

    def __len__(self):

        return len(self.item_list)

    def __read_json(self, root_dir, json_path):
        item_list = []
        for item in JsonHelper.load_file(json_path):
            img_path = os.path.join(root_dir, item['image_path'])
            if not os.path.exists(img_path) or not ImageHelper.is_img(img_path):
                Log.error('Image Path: {} is Invalid.'.format(img_path))
                exit(1)

            item_list.append((img_path, '.'.join(item['image_path'].split('.')[:-1])))

        Log.info('There are {} images..'.format(len(item_list)))
        return item_list
