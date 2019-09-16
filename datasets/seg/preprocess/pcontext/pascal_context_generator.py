#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhen Zhu(zzhu@hust.edu.cn)
# Generate train & val data.


import os
import argparse
import shutil
from PIL import Image
import numpy as np


LABEL_DIR = 'label'
IMAGE_DIR = 'image'


class PascalContextGenerator(object):
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

    def _class_to_index(self, mask, _mapping, _key):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert (values[i] in _mapping)
        index = np.digitize(mask.ravel(), _mapping, right=True)
        return _key[index].reshape(mask.shape)

    def generate_label(self):
        _image_dir = os.path.join(self.args.ori_root_dir, 'JPEGImages')
        annFile = os.path.join(self.args.ori_root_dir, 'trainval_merged.json')
        _mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296,
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424,
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360,
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
        _key = np.array(range(len(_mapping))).astype('uint8')

        from detail import Detail
        train_detail = Detail(annFile, _image_dir, 'train')
        train_ids = train_detail.getImgs()
        for img_id in train_ids:
            mask = Image.fromarray(self._class_to_index(
                train_detail.getMask(img_id), _mapping=_mapping, _key=_key))
            filename = img_id['file_name']
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(_image_dir, filename)
                shutil.copy(imgpath,
                            os.path.join(self.train_image_dir, filename))
                mask_png_name = basename + '.png'
                mask.save(os.path.join(self.train_label_dir, mask_png_name))

        val_detail = Detail(annFile, _image_dir, 'val')
        val_ids = val_detail.getImgs()
        for img_id in val_ids:
            mask = Image.fromarray(self._class_to_index(
                val_detail.getMask(img_id), _mapping=_mapping, _key=_key))
            filename = img_id['file_name']
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(_image_dir, filename)
                shutil.copy(imgpath,
                            os.path.join(self.val_image_dir, filename))
                mask_png_name = basename + '.png'
                mask.save(os.path.join(self.val_label_dir, mask_png_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    # ori_root_dir: VOCdevkit/VOC2010
    parser.add_argument('--ori_root_dir', default=None, type=str,
                        dest='ori_root_dir', help='The directory of the cityscapes data.')

    args = parser.parse_args()

    pascalcontext_seg_generator = PascalContextGenerator(args)
    pascalcontext_seg_generator.generate_label()
