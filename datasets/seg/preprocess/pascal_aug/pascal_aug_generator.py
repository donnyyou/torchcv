#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# COCO det data generator.


import os
import argparse
import shutil
from tqdm import tqdm
import scipy.io
from PIL import Image


LABEL_DIR = 'label'
IMAGE_DIR = 'image'


class PascalAugSegGenerator(object):
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

    def _load_mat(self, filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True,
            struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def generate_label(self):
        _mask_dir = os.path.join(self.args.ori_root_dir, 'cls')
        _image_dir = os.path.join(self.args.ori_root_dir, 'img')
        # train/val/test splits are pre-cut
        _splits_dir = self.args.ori_root_dir

        train_imgs = []

        _split_f_train = os.path.join(_splits_dir, 'train.txt')
        with open(os.path.join(_split_f_train), "r") as lines:
            for line in tqdm(lines):
                train_imgs.append(line.rstrip('\n') + ".jpg")

        for filename in train_imgs:
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(_image_dir, filename)
                maskname = basename + '.mat'
                maskpath = os.path.join(_mask_dir, maskname)
                if os.path.isfile(maskpath):
                    shutil.copy(imgpath,
                                os.path.join(self.train_image_dir, filename))
                    mask = self._load_mat(maskpath)
                    mask_png_name = basename + '.png'
                    mask.save(os.path.join(self.train_label_dir, mask_png_name))
                else:
                    print('cannot find the mask:', maskpath)

        val_imgs = []

        _split_f_val = os.path.join(_splits_dir, 'val.txt')
        with open(os.path.join(_split_f_val), "r") as lines:
            for line in tqdm(lines):
                val_imgs.append(line.rstrip('\n') + ".jpg")

        for filename in val_imgs:
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(_image_dir, filename)
                maskname = basename + '.mat'
                maskpath = os.path.join(_mask_dir, maskname)
                if os.path.isfile(maskpath):
                    shutil.copy(imgpath,
                                os.path.join(self.val_image_dir, filename))
                    mask = self._load_mat(maskpath)
                    mask_png_name = basename + '.png'
                    mask.save(os.path.join(self.val_label_dir, mask_png_name))
                else:
                    print('cannot find the mask:', maskpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    # ori_root_dir: VOCaug/dataset
    parser.add_argument('--ori_root_dir', default=None, type=str,
                        dest='ori_root_dir', help='The directory of the cityscapes data.')

    args = parser.parse_args()

    pascalaug_seg_generator = PascalAugSegGenerator(args)
    pascalaug_seg_generator.generate_label()
