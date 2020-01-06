#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Single Shot Detector data loader


import os
import torch
import numpy as np
import torch.utils.data as data

from lib.parallel.data_container import DataContainer
from lib.tools.helper.json_helper import JsonHelper
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as Log

# torch.multiprocessing.set_sharing_strategy('file_system')


class DefaultDataset(data.Dataset):

    def __init__(self, root_dir=None, dataset=None,
                 aug_transform=None, img_transform=None, configer=None):
        super(DefaultDataset, self).__init__()
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.img_list, self.json_list = self.__list_dirs(root_dir, dataset)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        bboxes, labels = self.__read_json_file(self.json_list[index])
        ori_bboxes, ori_labels = bboxes.copy(), labels.copy()

        if self.aug_transform is not None:
            img, bboxes, labels = self.aug_transform(img, bboxes=bboxes, labels=labels)

        labels = torch.from_numpy(labels).long()
        bboxes = torch.from_numpy(bboxes).float()

        meta = dict(
            ori_img_size=img_size,
            border_wh=ImageHelper.get_size(img),
            ori_bboxes=torch.from_numpy(ori_bboxes).float(),
            ori_labels=torch.from_numpy(ori_labels).long()
        )
        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            img=DataContainer(img, stack=True, return_dc=True, samples_per_gpu=True),
            bboxes=DataContainer(bboxes, stack=False, return_dc=True, samples_per_gpu=True),
            labels=DataContainer(labels, stack=False, return_dc=True, samples_per_gpu=True),
            meta=DataContainer(meta, stack=False, cpu_only=True, return_dc=True, samples_per_gpu=True)
        )

    def __len__(self):

        return len(self.img_list)

    def __read_json_file(self, json_file):
        """
            filename: JSON file

            return: three list: key_points list, centers list and scales list.
        """
        json_dict = JsonHelper.load_file(json_file)

        labels = list()
        bboxes = list()

        for object in json_dict['objects']:
            if 'difficult' in object and object['difficult'] and not self.configer.get('data', 'keep_difficult'):
                continue

            labels.append(object['label'])
            bboxes.append(object['bbox'])

        return np.array(bboxes).astype(np.float32), np.array(labels)

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        json_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        json_dir = os.path.join(root_dir, dataset, 'json')

        for file_name in os.listdir(json_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = ImageHelper.imgpath(image_dir, image_name)
            json_path = os.path.join(json_dir, file_name)
            if not os.path.exists(json_path) or img_path is None:
                Log.warn('Json Path: {} not exists.'.format(json_path))
                continue

            json_list.append(json_path)
            img_list.append(img_path)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            image_dir = os.path.join(root_dir, 'val/image')
            json_dir = os.path.join(root_dir, 'val/json')
            for file_name in os.listdir(json_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = ImageHelper.imgpath(image_dir, image_name)
                json_path = os.path.join(json_dir, file_name)
                if not os.path.exists(json_path) or img_path is None:
                    Log.warn('Json Path: {} not exists.'.format(json_path))
                    continue

                json_list.append(json_path)
                img_list.append(img_path)

        return img_list, json_list
