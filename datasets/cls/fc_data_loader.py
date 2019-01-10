#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Data loader for Image Classification.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import torch.utils.data as data

from extensions.parallel.data_container import DataContainer
from utils.helpers.image_helper import ImageHelper


class FCDataLoader(data.Dataset):

    def __init__(self, root_dir=None, aug_transform=None, img_transform=None, configer=None):

        self.img_list, self.label_list = self.__read_json_file(root_dir)
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.configer = configer

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        label = self.label_list[index]

        if self.aug_transform is not None:
            img = self.aug_transform(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            img=DataContainer(img, stack=True),
            label=DataContainer(label, stack=True),
        )

    def __len__(self):

        return len(self.img_list)

    def __read_json_file(self, root_dir):
        img_list = list()
        label_list = list()

        with open(os.path.join(root_dir, 'label.json'), 'r') as file_stream:
            items = json.load(file_stream)
            for item in items:
                img_list.append(os.path.join(root_dir, item['image_path']))
                label_list.append(item['label'])

        return img_list, label_list


if __name__ == "__main__":
    # Test data loader.
    pass