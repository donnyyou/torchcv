#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Single Shot Detector data loader


import os
import torch.utils.data as data

from lib.parallel.data_container import DataContainer
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as Log


class DefaultPix2pixDataset(data.Dataset):

    def __init__(self, root_dir=None, dataset=None,
                 aug_transform=None, img_transform=None, configer=None):
        super(DefaultPix2pixDataset, self).__init__()
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.imgA_list, self.imgB_list = self.__list_dirs(root_dir, dataset)

    def __getitem__(self, index):
        imgA = ImageHelper.read_image(self.imgA_list[index],
                                      tool=self.configer.get('data', 'image_tool'),
                                      mode=self.configer.get('data', 'input_mode'))

        imgB = ImageHelper.read_image(self.imgB_list[index],
                                      tool=self.configer.get('data', 'image_tool'),
                                      mode=self.configer.get('data', 'input_mode'))

        if self.aug_transform is not None:
            imgA, imgB = self.aug_transform([imgA, imgB])

        if self.img_transform is not None:
            imgA = self.img_transform(imgA)
            imgB = self.img_transform(imgB)


        return dict(
            imgA=DataContainer(imgA, stack=True),
            imgB=DataContainer(imgB, stack=True),
        )

    def __len__(self):

        return len(self.imgA_list)

    def __list_dirs(self, root_dir, dataset):
        imgA_list = list()
        imgB_list = list()

        imageA_dir = os.path.join(root_dir, dataset, 'imageA')
        imageB_dir = os.path.join(root_dir, dataset, 'imageB')

        for file_name in os.listdir(imageA_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            imgA_path = ImageHelper.imgpath(imageA_dir, image_name)
            imgB_path = ImageHelper.imgpath(imageB_dir, image_name)
            if not os.path.exists(imgA_path) or not os.path.exists(imgB_path):
                Log.warn('Img Path: {} not exists.'.format(imgA_path))
                continue

            imgA_list.append(imgA_path)
            imgB_list.append(imgB_path)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            imageA_dir = os.path.join(root_dir, 'val/imageA')
            imageB_dir = os.path.join(root_dir, 'val/imageB')
            for file_name in os.listdir(imageA_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                imgA_path = ImageHelper.imgpath(imageA_dir, image_name)
                imgB_path = ImageHelper.imgpath(imageB_dir, image_name)
                if not os.path.exists(imgA_path) or not os.path.exists(imgB_path):
                    Log.warn('Img Path: {} not exists.'.format(imgA_path))
                    continue

                imgA_list.append(imgA_path)
                imgB_list.append(imgB_path)

        return imgA_list, imgB_list
