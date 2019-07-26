#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# KITTI det data generator.


import json
import os
import argparse
import shutil


JOSN_DIR = 'json'
IMAGE_DIR = 'image'
CAT_DICT = {'Car': 0, }


class KittiDetGenerator(object):

    def __init__(self, args, json_dir=JOSN_DIR, image_dir=IMAGE_DIR):
        self.args = args
        self.train_json_dir = os.path.join(self.args.save_dir, 'train', json_dir)
        self.val_json_dir = os.path.join(self.args.save_dir, 'val', json_dir)
        if not os.path.exists(self.train_json_dir):
            os.makedirs(self.train_json_dir)

        if not os.path.exists(self.val_json_dir):
            os.makedirs(self.val_json_dir)

        self.train_image_dir = os.path.join(self.args.save_dir, 'train', image_dir)
        self.val_image_dir = os.path.join(self.args.save_dir, 'val', image_dir)
        if not os.path.exists(self.train_image_dir):
            os.makedirs(self.train_image_dir)

        if not os.path.exists(self.val_image_dir):
            os.makedirs(self.val_image_dir)

    def generate_label(self):
        file_count = 0
        for label_file in os.listdir(self.args.ori_label_dir):
            file_count += 1
            label_file_path = os.path.join(self.args.ori_label_dir, label_file)
            object_list = list()
            tree_dict = dict()
            with open(label_file_path, 'r') as file_stream:
                for line in file_stream.readlines():
                    object = dict()
                    line = line.split()
                    name = line[0]
                    if name in CAT_DICT:
                        # name bbox key_points alpha.
                        data = [float(x) for x in line[1:]]
                        box = data[3:7]
                        object['bbox'] = box
                        object['label'] = CAT_DICT[name]

                        object_list.append(object)

            tree_dict['objects'] = object_list
            if len(object_list) == 0:
                continue

            if file_count % self.args.val_interval == 0:
                fw = open(os.path.join(self.val_json_dir, '{}.json'.format(label_file.split('.')[0])), 'w')
                fw.write(json.dumps(tree_dict))
                fw.close()
                shutil.copy(os.path.join(self.args.ori_img_dir, '{}.png'.format(label_file.split('.')[0])),
                            os.path.join(self.val_image_dir, '{}.png'.format(label_file.split('.')[0])))
            else:
                fw = open(os.path.join(self.train_json_dir, '{}.json'.format(label_file.split('.')[0])), 'w')
                fw.write(json.dumps(tree_dict))
                fw.close()
                shutil.copy(os.path.join(self.args.ori_img_dir, '{}.png'.format(label_file.split('.')[0])),
                            os.path.join(self.train_image_dir, '{}.png'.format(label_file.split('.')[0])))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_img_dir', default=None, type=str,
                        dest='ori_img_dir', help='The directory of the image data.')
    parser.add_argument('--ori_label_dir', default=None, type=str,
                        dest='ori_label_dir', help='The directory of the label data.')
    parser.add_argument('--val_interval', default=10, type=float,
                        dest='val_interval', help='The ratio of train & val data.')

    args = parser.parse_args()

    kitti_det_generator = KittiDetGenerator(args)
    kitti_det_generator.generate_label()
