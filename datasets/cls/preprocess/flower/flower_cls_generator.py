#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# KITTI det data generator.


import json
import os
import argparse
import shutil


IMAGE_DIR = 'image'


class FlowerClsGenerator(object):

    def __init__(self, args, image_dir=IMAGE_DIR):
        self.args = args
        self.train_json_file = os.path.join(self.args.save_dir, 'train/label.json')
        self.val_json_file = os.path.join(self.args.save_dir, 'val/label.json')
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        self.train_image_dir = os.path.join(self.args.save_dir, 'train', image_dir)
        self.val_image_dir = os.path.join(self.args.save_dir, 'val', image_dir)
        if not os.path.exists(self.train_image_dir):
            os.makedirs(self.train_image_dir)

        if not os.path.exists(self.val_image_dir):
            os.makedirs(self.val_image_dir)

    def generate_label(self):
        train_json_list = list()
        val_json_list = list()

        img_count = 0
        for image_file in os.listdir(self.args.ori_img_dir):
            shotname, extension = os.path.splitext(image_file)
            if extension != '.jpg':
                continue

            img_count += 1
            img_dict = dict()
            img_dict['image_path'] = '{}/{}'.format(IMAGE_DIR, image_file)
            img_dict['label'] = (int(shotname.split('_')[-1]) - 1) // 80

            if img_count % self.args.val_interval == 0:
                val_json_list.append(img_dict)
                shutil.copy(os.path.join(self.args.ori_img_dir, image_file),
                            os.path.join(self.val_image_dir, image_file))
            else:
                train_json_list.append(img_dict)
                shutil.copy(os.path.join(self.args.ori_img_dir, image_file),
                            os.path.join(self.train_image_dir, image_file))

        fw = open(self.train_json_file, 'w')
        fw.write(json.dumps(train_json_list))
        fw.close()

        fw = open(self.val_json_file, 'w')
        fw.write(json.dumps(val_json_list))
        fw.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_img_dir', default=None, type=str,
                        dest='ori_img_dir', help='The directory of the image data.')
    parser.add_argument('--val_interval', default=10, type=float,
                        dest='val_interval', help='The ratio of train & val data.')

    args = parser.parse_args()

    flower_cls_generator = FlowerClsGenerator(args)
    flower_cls_generator.generate_label()
