#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Single Shot Detector data loader


import os
import json
import random
import torch.utils.data as data

from extensions.tools.parallel import DataContainer
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class FaceGANLoader(data.Dataset):

    def __init__(self, root_dir=None, dataset=None,
                 aug_transform=None, img_transform=None, configer=None):
        super(FaceGANLoader, self).__init__()
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.imgA_list, self.labelA_list, self.imgB_list, self.labelB_list = self.__read_json_file(root_dir, dataset)

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

    def __read_json_file(self, root_dir, dataset):
        imgA_list = list()
        imgB_list = list()
        labelA_list = list()
        labelB_list = list()

        with open(os.path.join(root_dir, dataset, 'labelA.json'), 'r') as file_stream:
            items = json.load(file_stream)
            for item in items:
                img_path = os.path.join(root_dir, dataset, item['image_path'])
                if not os.path.exists(img_path):
                    Log.warn('Image Path: {} not exists.'.format(img_path))
                    continue

                imgA_list.append(img_path)
                labelA_list.append(item['label'])

        with open(os.path.join(root_dir, dataset, 'labelB.json'), 'r') as file_stream:
            items = json.load(file_stream)
            for item in items:
                img_path = os.path.join(root_dir, dataset, item['image_path'])
                if not os.path.exists(img_path):
                    Log.warn('Image Path: {} not exists.'.format(img_path))
                    continue

                imgB_list.append(img_path)
                labelB_list.append(item['label'])

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            with open(os.path.join(root_dir, 'val', 'labelA.json'), 'r') as file_stream:
                items = json.load(file_stream)
                for item in items:
                    img_path = os.path.join(root_dir, 'val', item['image_path'])
                    if not os.path.exists(img_path):
                        Log.warn('Image Path: {} not exists.'.format(img_path))
                        continue

                    imgA_list.append(img_path)
                    labelA_list.append(item['label'])

            with open(os.path.join(root_dir, 'val', 'labelB.json'), 'r') as file_stream:
                items = json.load(file_stream)
                for item in items:
                    img_path = os.path.join(root_dir, 'val', item['image_path'])
                    if not os.path.exists(img_path):
                        Log.warn('Image Path: {} not exists.'.format(img_path))
                        continue

                    imgB_list.append(img_path)
                    labelB_list.append(item['label'])

        return imgA_list, labelA_list, imgB_list, labelB_list

    def __list_dirs(self, root_dir, dataset):
        imgA_list = list()
        imgB_list = list()
        imageA_dir = os.path.join(root_dir, dataset, 'imageA')
        imageB_dir = os.path.join(root_dir, dataset, 'imageB')

        for file_name in os.listdir(imageA_dir):
            imgA_path = os.path.join(imageA_dir, file_name)
            imgA_list.append(imgA_path)

        for file_name in os.listdir(imageB_dir):
            imgB_path = os.path.join(imageB_dir, file_name)
            imgB_list.append(imgB_path)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            imageA_dir = os.path.join(root_dir, 'val/imageA')
            imageB_dir = os.path.join(root_dir, 'val/imageB')

            for file_name in os.listdir(imageA_dir):
                imgA_path = os.path.join(imageA_dir, file_name)
                imgA_list.append(imgA_path)

            for file_name in os.listdir(imageB_dir):
                imgB_path = os.path.join(imageB_dir, file_name)
                imgB_list.append(imgB_path)

        return imgA_list, imgB_list