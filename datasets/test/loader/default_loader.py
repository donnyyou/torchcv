#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Single Shot Detector data loader


import os
import torch.utils.data as data

from extensions.tools.parallel import DataContainer
from utils.helpers.file_helper import FileHelper
from utils.helpers.image_helper import ImageHelper


class DefaultLoader(data.Dataset):

    def __init__(self, test_dir=None, img_transform=None, configer=None):
        super(DefaultLoader, self).__init__()
        self.configer = configer
        self.img_transform = img_transform
        self.img_list = [os.path.join(test_dir, filename) for filename in FileHelper.list_dir(test_dir)]

    def __getitem__(self, index):
        image = ImageHelper.read_image(self.img_list[index],
                                       tool=self.configer.get('data', 'image_tool'),
                                       mode=self.configer.get('data', 'input_mode'))

        img_size = ImageHelper.get_size(image)
        if self.configer.exists('test', 'input_size'):
            input_size = self.configer.get('test', 'input_size')
            if input_size[0] == -1 and input_size[1] == -1:
                in_width, in_height = ImageHelper.get_size(image)

            elif input_size[0] != -1 and input_size[1] != -1:
                in_width, in_height = input_size

            elif input_size[0] == -1 and input_size[1] != -1:
                width, height = ImageHelper.get_size(image)
                scale_ratio = input_size[1] / height
                w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
                in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

            else:
                assert input_size[0] != -1 and input_size[1] == -1
                width, height = ImageHelper.get_size(image)
                scale_ratio = input_size[0] / width
                w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
                in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

        elif self.configer.exists('test', 'min_side_length') and not self.configer.exists('test', 'max_side_length'):
            width, height = ImageHelper.get_size(image)
            scale_ratio = self.configer.get('test', 'min_side_length') / min(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

        elif not self.configer.exists('test', 'min_side_length') and self.configer.exists('test', 'max_side_length'):
            width, height = ImageHelper.get_size(image)
            scale_ratio = self.configer.get('test', 'max_side_length') / max(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

        elif self.configer.exists('test', 'min_side_length') and self.configer.exists('test', 'max_side_length'):
            width, height = ImageHelper.get_size(image)
            scale_ratio = self.configer.get('test', 'min_side_length') / min(width, height)
            bound_scale_ratio = self.configer.get('test', 'max_side_length') / max(width, height)
            scale_ratio = min(scale_ratio, bound_scale_ratio)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

        else:
            in_width, in_height = ImageHelper.get_size(image)

        img = ImageHelper.resize(image, (int(in_width), int(in_height)), interpolation='cubic')
        if self.img_transform is not None:
            img = self.img_transform(img)

        meta = dict(
            ori_img_size=img_size,
            border_size=[in_width, in_height],
            img_path=self.img_list[index]
        )
        return dict(
            img=DataContainer(img, stack=True, return_dc=True, samples_per_gpu=True),
            meta=DataContainer(meta, stack=False, cpu_only=True, return_dc=True, samples_per_gpu=True)
        )

    def __len__(self):

        return len(self.img_list)
