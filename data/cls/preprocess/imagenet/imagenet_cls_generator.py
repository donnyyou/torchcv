#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# KITTI det data generator.


import json
import os
import argparse
import shutil


NUM_OF_CLASSES = 1000


class ImageNetClsGenerator(object):

    def __init__(self, args):
        self.args = args
        self.train_json_file = os.path.join(self.args.save_dir, 'train.json')
        self.val_json_file = os.path.join(self.args.save_dir, 'val.json')
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

    def generate_label(self):
        train_json_list = list()
        val_json_list = list()

        with open('imagenet_class_index.json', 'r') as imagenet_stream:
            imagenet_class_dict = json.load(imagenet_stream)

        img_count = 0
        for i in range(NUM_OF_CLASSES):
            train_folder = os.path.join(self.args.ori_img_dir, 'train', imagenet_class_dict[str(i)][0])
            val_folder = os.path.join(self.args.ori_img_dir, 'val', imagenet_class_dict[str(i)][0])

            for image_file in os.listdir(train_folder):
                img_count += 1
                img_dict = dict()
                img_dict['image_path'] = 'train/{}/{}'.format(imagenet_class_dict[str(i)][0], image_file)
                img_dict['label'] = i

                train_json_list.append(img_dict)

            for image_file in os.listdir(val_folder):
                img_count += 1
                img_dict = dict()
                img_dict['image_path'] = 'val/{}/{}'.format(imagenet_class_dict[str(i)][0], image_file)
                img_dict['label'] = i

                val_json_list.append(img_dict)

        fw = open(self.train_json_file, 'w')
        fw.write(json.dumps(train_json_list))
        fw.close()

        fw = open(self.val_json_file, 'w')
        fw.write(json.dumps(val_json_list))
        fw.close()
        shutil.move(self.args.ori_img_dir, os.path.join(self.args.save_dir, 'dataset'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_img_dir', default=None, type=str,
                        dest='ori_img_dir', help='The directory of the image data.')

    args = parser.parse_args()

    imagenet_cls_generator = ImageNetClsGenerator(args)
    imagenet_cls_generator.generate_label()
