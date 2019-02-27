#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# COCO det data generator.


import os
import argparse
import shutil


LABEL_DIR = 'label'
IMAGE_DIR = 'image'


class ADE20KSegGenerator(object):
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
        train_img_folder = os.path.join(self.args.ori_root_dir, 'images/training')
        train_mask_folder = os.path.join(self.args.ori_root_dir, 'annotations/training')

        val_img_folder = os.path.join(self.args.ori_root_dir, 'images/validation')
        val_mask_folder = os.path.join(self.args.ori_root_dir, 'annotations/validation')

        for filename in os.listdir(train_img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(train_img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(train_mask_folder, maskname)
                if os.path.isfile(maskpath):
                    shutil.copy(imgpath,
                                os.path.join(self.train_image_dir, filename))
                    shutil.copy(maskpath,
                                os.path.join(self.train_label_dir, maskname))
                else:
                    print('cannot find the mask:', maskpath)

        for filename in os.listdir(val_img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(val_img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(val_mask_folder, maskname)
                if os.path.isfile(maskpath):
                    shutil.copy(imgpath, os.path.join(self.val_image_dir, filename))
                    shutil.copy(maskpath, os.path.join(self.val_label_dir, maskname))
                else:
                    print('cannot find the mask:', maskpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_root_dir', default=None, type=str,
                        dest='ori_root_dir', help='The directory of the cityscapes data.')

    args = parser.parse_args()

    ade20k_seg_generator = ADE20KSegGenerator(args)
    ade20k_seg_generator.generate_label()
