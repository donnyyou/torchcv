#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Single Shot Detector data loader


import os
import torch.utils.data as data

from extensions.tools.parallel import DataContainer
from utils.helpers.file_helper import FileHelper
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class DefaultLoader(data.Dataset):

    def __init__(self, test_dir=None, list_path=None, aug_transform=None, img_transform=None, configer=None):
        super(DefaultLoader, self).__init__()
        self.configer = configer
        self.aug_transform=aug_transform
        self.img_transform = img_transform
        if test_dir is not None:
            self.img_list = [os.path.join(test_dir, filename)
                             for filename in FileHelper.list_dir(test_dir) if ImageHelper.is_img(filename)]
        else:
            self.img_list = self.__read_list(list_path)

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

    def __read_list(self, list_path):
        img_list = []
        with open(list_path, 'r') as f:
            for line in f.readlines()[0:]:
                img_path = line.strip().split()[0]
                if not os.path.exists(img_path):
                    Log.error('Image Path: {} is Invalid.'.format(img_path))
                    exit(1)

                img_list.append(img_path)

        Log.info('There are {} images..'.format(len(img_list)))
        return img_list
