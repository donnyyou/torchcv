#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import os
import numpy as np
from torch.utils import data

from lib.parallel.data_container import DataContainer
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.helper.file_helper import FileHelper
from lib.tools.util.logger import Logger as Log


class CityscapesDataset(data.Dataset):
    def __init__(self, root_dir, dataset=None, aug_transform=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_list, self.label_list = self.__list_dirs(root_dir, dataset)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        labelmap = ImageHelper.read_image(self.label_list[index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')
        if self.configer.get('data.label_list', default=None):
            labelmap = self._encode_label(labelmap)

        if self.configer.get('data.reduce_zero_label', default=None):
            labelmap = self._reduce_zero_label(labelmap)

        ori_target = ImageHelper.to_np(labelmap)

        if self.aug_transform is not None:
            img, labelmap = self.aug_transform(img, labelmap=labelmap)

        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)

        meta = dict(
            ori_img_wh=img_size,
            border_wh=border_size,
            ori_target=ori_target
        )
        return dict(
            img=DataContainer(img, stack=True),
            labelmap=DataContainer(labelmap, stack=True),
            meta=DataContainer(meta, stack=False, cpu_only=True),
        )

    def _reduce_zero_label(self, labelmap):
        if not self.configer.get('data', 'reduce_zero_label'):
            return labelmap

        labelmap = np.array(labelmap)
        labelmap[labelmap == 0] = 255
        labelmap = labelmap - 1
        labelmap[labelmap == 254] = 255
        if self.configer.get('data', 'image_tool') == 'pil':
            labelmap = ImageHelper.to_img(labelmap.astype(np.uint8))

        return labelmap

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)
        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i in range(len(self.configer.get('data', 'label_list'))):
            class_id = self.configer.get('data', 'label_list')[i]
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.to_img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        image_dir = os.path.join(root_dir, 'leftImg8bit', dataset)
        label_dir = os.path.join(root_dir, 'gtFine', dataset)

        for image_file in FileHelper.list_dir(image_dir):
            image_name = '_'.join(image_file.split('_')[:-1])
            label_file = '{}_gtFine_labelIds.png'.format(image_name)
            img_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, label_file)
            if not (os.path.exists(label_path) and os.path.exists(img_path)):
                Log.warn('Image/Label Path: {} not exists.'.format(image_name))
                continue

            img_list.append(img_path)
            label_list.append(label_path)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            image_dir = os.path.join(root_dir, 'leftImg8bit/val')
            label_dir = os.path.join(root_dir, 'gtFine/val')
    
            for image_file in FileHelper.list_dir(image_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                label_file = '{}_gtFine_labelIds.png'.format(image_name)
                img_path = os.path.join(image_dir, image_file)
                label_path = os.path.join(label_dir, label_file)
                if not (os.path.exists(label_path) and os.path.exists(img_path)):
                    Log.warn('Image/Label Path: {} not exists.'.format(image_name))
                    continue
    
                img_list.append(img_path)
                label_list.append(label_path)


        return img_list, label_list


if __name__ == "__main__":
    # Test cityscapes loader.
    pass
