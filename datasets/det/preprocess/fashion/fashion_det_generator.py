#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Fashion det data generator.


import json
import os
import argparse
import shutil


JOSN_DIR = 'json'
IMAGE_DIR = 'image'
CAT_DICT = {
    'bags': 0, 'belts': 1, 'dresses': 2, 'eyewear': 3, 'footwear': 4, 'hats': 5,
    'leggings': 6, 'outerwear': 7, 'pants': 8, 'skirts': 9, 'tops': 10
}


class FashionDetGenerator(object):

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
        type_stream = open(self.args.ori_type_file, 'r')
        bbox_stream = open(self.args.ori_bbox_file, 'r')
        type_lines = type_stream.readlines()
        bbox_lines = bbox_stream.readlines()
        assert len(type_lines) == len(bbox_lines)
        file_count = 0
        max_cat = 0
        for (type_line, bbox_line) in zip(type_lines, bbox_lines):
            if 'jpg' not in type_line or 'jpg' not in bbox_line:
                print(type_line)
                print(bbox_line)
                continue

            type_line_items = type_line.split()
            bbox_line_items = bbox_line.split()
            assert type_line_items[0] == bbox_line_items[0]
            file_count += 1
            json_dict = dict()
            object_list = list()
            object_dict = dict()
            object_dict['bbox'] = [float(bbox_line_items[1]), float(bbox_line_items[2]),
                                   float(bbox_line_items[3]), float(bbox_line_items[4])]

            object_dict['label'] = int(type_line_items[1]) - 1
            max_cat = int(type_line_items[1]) if int(type_line_items[1]) > max_cat else max_cat
            object_list.append(object_dict)
            json_dict['objects'] = object_list
            file_name = type_line_items[0].rstrip().split('/')[-1]
            if file_count % self.args.val_interval == 0:
                fw = open(os.path.join(self.val_json_dir, '{}.json'.format(file_name.split('.')[0])), 'w')
                fw.write(json.dumps(json_dict))
                fw.close()
                shutil.copy(os.path.join(self.args.ori_img_dir, file_name),
                            os.path.join(self.val_image_dir, file_name))
            else:
                fw = open(os.path.join(self.train_json_dir, '{}.json'.format(file_name.split('.')[0])), 'w')
                fw.write(json.dumps(json_dict))
                fw.close()
                shutil.copy(os.path.join(self.args.ori_img_dir, file_name),
                            os.path.join(self.train_image_dir, file_name))

            print (max_cat)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_img_dir', default=None, type=str,
                        dest='ori_img_dir', help='The directory of the image data.')
    parser.add_argument('--ori_type_file', default=None, type=str,
                        dest='ori_type_file', help='The directory of the type data.')
    parser.add_argument('--ori_bbox_file', default=None, type=str,
                        dest='ori_bbox_file', help='The directory of the bbox data.')
    parser.add_argument('--val_interval', default=10, type=float,
                        dest='val_interval', help='The ratio of train & val data.')

    args = parser.parse_args()

    fashion_det_generator = FashionDetGenerator(args)
    fashion_det_generator.generate_label()
