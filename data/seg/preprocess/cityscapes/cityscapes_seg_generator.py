#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# CityScape Seg data generator.


import os
import argparse
import shutil


IMAGE_DIR = 'image'
LABEL_DIR = 'label'

def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class CityscapesSegGenerator(object):

    def __init__(self, args, image_dir=IMAGE_DIR, label_dir=LABEL_DIR):
        self.args = args
        self.train_label_dir = os.path.join(self.args.save_dir, 'train', label_dir)
        self.val_label_dir = os.path.join(self.args.save_dir, 'val', label_dir)
        if not os.path.exists(self.train_label_dir):
            os.makedirs(self.train_label_dir)

        if not os.path.exists(self.val_label_dir):
            os.makedirs(self.val_label_dir)

        self.train_image_dir = os.path.join(self.args.save_dir, 'train', image_dir)
        self.val_image_dir = os.path.join(self.args.save_dir, 'val', image_dir)
        if not os.path.exists(self.train_image_dir):
            os.makedirs(self.train_image_dir)

        if not os.path.exists(self.val_image_dir):
            os.makedirs(self.val_image_dir)

    def generate_label(self):
        if not self.args.coarse:
            ori_train_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/train')
            ori_train_label_dir = os.path.join(self.args.ori_root_dir, 'gtFine/train')
            ori_val_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/val')
            ori_val_label_dir = os.path.join(self.args.ori_root_dir, 'gtFine/val')

            for image_file in self.__list_dir(ori_train_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                label_file = '{}_gtFine_labelIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_train_img_dir, image_file),
                            os.path.join(self.train_image_dir, '{}{}'.format(shotname, extension)))
                shutil.copy(os.path.join(ori_train_label_dir, label_file),
                            os.path.join(self.train_label_dir, '{}.png'.format(shotname)))

            for image_file in self.__list_dir(ori_val_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                label_file = '{}_gtFine_labelIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_val_img_dir, image_file),
                            os.path.join(self.val_image_dir, '{}{}'.format(shotname, extension)))
                shutil.copy(os.path.join(ori_val_label_dir, label_file),
                            os.path.join(self.val_label_dir, '{}.png'.format(shotname)))

        else:

            ori_train_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/train')
            ori_train_label_dir = os.path.join(self.args.ori_root_dir, 'gtCoarse/train')
            ori_train_extra_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/train_extra')
            ori_train_extra_label_dir = os.path.join(self.args.ori_root_dir, 'gtCoarse/train_extra')
            ori_val_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/val')
            ori_val_label_dir = os.path.join(self.args.ori_root_dir, 'gtCoarse/val')

            for image_file in self.__list_dir(ori_train_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                label_file = '{}_gtCoarse_labelIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_train_img_dir, image_file),
                            os.path.join(self.train_image_dir, '{}{}'.format(shotname, extension)))
                shutil.copy(os.path.join(ori_train_label_dir, label_file),
                            os.path.join(self.train_label_dir, '{}.png'.format(shotname)))

            for image_file in self.__list_dir(ori_train_extra_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                label_file = '{}_gtCoarse_labelIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_train_extra_img_dir, image_file),
                            os.path.join(self.train_image_dir, '{}{}'.format(shotname, extension)))
                shutil.copy(os.path.join(ori_train_extra_label_dir, label_file),
                            os.path.join(self.train_label_dir, '{}.png'.format(shotname)))

            for image_file in self.__list_dir(ori_val_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                label_file = '{}_gtCoarse_labelIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_val_img_dir, image_file),
                            os.path.join(self.val_image_dir, '{}{}'.format(shotname, extension)))
                shutil.copy(os.path.join(ori_val_label_dir, label_file),
                            os.path.join(self.val_label_dir, '{}.png'.format(shotname)))


    def __list_dir(self, dir_name):
        filename_list = list()
        for item in os.listdir(dir_name):
            if os.path.isdir(os.path.join(dir_name, item)):
                for filename in os.listdir(os.path.join(dir_name, item)):
                    filename_list.append('{}/{}'.format(item, filename))
            else:
                filename_list.append(item)

        return filename_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse', type=str2bool, nargs='?', default=False,
                        dest='coarse', help='Whether is the coarse data.')
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_root_dir', default=None, type=str,
                        dest='ori_root_dir', help='The directory of the cityscapes data.')

    args = parser.parse_args()

    cityscapes_seg_generator = CityscapesSegGenerator(args)
    cityscapes_seg_generator.generate_label()
