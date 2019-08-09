#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# VOC det data generator.


import json
import os
import argparse
import shutil
from bs4 import BeautifulSoup


JOSN_DIR = 'json'
IMAGE_DIR = 'image'
CAT_DICT = {
    'aeroplane': 0, 'bicycle':1,'bird':2,'boat':3,'bottle':4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
    'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}


class VocDetGenerator(object):

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

    def _get_info_tree(self, label_file, dataset='VOC2007'):
        label_file_path = os.path.join(self.args.root_dir, dataset, 'Annotations', label_file)
        object_list = list()
        tree_dict = dict()
        with open(label_file_path, 'r') as file_stream:
            xml_tree = file_stream.readlines()
            xml_tree = ''.join([line.strip('\t') for line in xml_tree])
            xml_tree = BeautifulSoup(xml_tree, "html5lib")
            for obj in xml_tree.findAll('object'):
                object = dict()
                for name_tag in obj.findChildren('name'):
                    name = str(name_tag.contents[0])
                    difficult = int(obj.find('difficult').contents[0])
                    if name in CAT_DICT:
                        bbox = obj.findChildren('bndbox')[0]
                        # 1-indexing to 0-indexing.
                        xmin = int(float(bbox.findChildren('xmin')[0].contents[0])) - 1
                        ymin = int(float(bbox.findChildren('ymin')[0].contents[0])) - 1
                        xmax = int(float(bbox.findChildren('xmax')[0].contents[0])) - 1
                        ymax = int(float(bbox.findChildren('ymax')[0].contents[0])) - 1
                        object['bbox'] = [xmin, ymin, xmax, ymax]
                        object['label'] = CAT_DICT[name]
                        object['difficult'] = difficult
                        object_list.append(object)

        tree_dict['objects'] = object_list
        return tree_dict

    def generate_label(self):
        file_count = 0
        if self.args.dataset in ['VOC07', 'VOC07+12', 'VOC07++12']:
            with open(os.path.join(self.args.root_dir, 'VOC2007/ImageSets/Main/trainval.txt'), 'r') as train_stream:
                for img_name in train_stream.readlines():
                    img_name = img_name.rstrip()
                    label_file = '{}.xml'.format(img_name)
                    file_count += 1
                    tree_dict = self._get_info_tree(label_file, dataset='VOC2007')
                    fw = open(os.path.join(self.train_json_dir, '{}.json'.format(img_name)), 'w')
                    fw.write(json.dumps(tree_dict))
                    fw.close()
                    shutil.copy(os.path.join(self.args.root_dir, 'VOC2007/JPEGImages', '{}.jpg'.format(img_name)),
                                os.path.join(self.train_image_dir, '{}.jpg'.format(img_name)))

        if self.args.dataset in ['VOC07+12', 'VOC07++12', 'VOC12']:
            with open(os.path.join(self.args.root_dir, 'VOC2012/ImageSets/Main/trainval.txt'), 'r') as train_stream:
                for img_name in train_stream.readlines():
                    img_name = img_name.rstrip()
                    label_file = '{}.xml'.format(img_name)
                    file_count += 1
                    tree_dict = self._get_info_tree(label_file, dataset='VOC2012')
                    fw = open(os.path.join(self.train_json_dir, '{}.json'.format(img_name)), 'w')
                    fw.write(json.dumps(tree_dict))
                    fw.close()
                    shutil.copy(os.path.join(self.args.root_dir, 'VOC2012/JPEGImages', '{}.jpg'.format(img_name)),
                                os.path.join(self.train_image_dir, '{}.jpg'.format(img_name)))

        if self.args.dataset in ['VOC07++12']:
            with open(os.path.join(self.args.root_dir, 'VOC2007/ImageSets/Main/test.txt'), 'r') as train_stream:
                for img_name in train_stream.readlines():
                    img_name = img_name.rstrip()
                    label_file = '{}.xml'.format(img_name)
                    file_count += 1
                    tree_dict = self._get_info_tree(label_file)
                    fw = open(os.path.join(self.train_json_dir, '{}.json'.format(img_name)), 'w')
                    fw.write(json.dumps(tree_dict))
                    fw.close()
                    shutil.copy(os.path.join(self.args.root_dir, 'VOC2007/JPEGImages', '{}.jpg'.format(img_name)),
                                os.path.join(self.train_image_dir, '{}.jpg'.format(img_name)))

        with open(os.path.join(self.args.root_dir, 'VOC2007/ImageSets/Main/test.txt'), 'r') as train_stream:
            for img_name in train_stream.readlines():
                img_name = img_name.rstrip()
                label_file = '{}.xml'.format(img_name)
                file_count += 1
                tree_dict = self._get_info_tree(label_file)
                fw = open(os.path.join(self.val_json_dir, '{}.json'.format(img_name)), 'w')
                fw.write(json.dumps(tree_dict))
                fw.close()
                shutil.copy(os.path.join(self.args.root_dir, 'VOC2007/JPEGImages', '{}.jpg'.format(img_name)),
                            os.path.join(self.val_image_dir, '{}.jpg'.format(img_name)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--root_dir', default=None, type=str,
                        dest='root_dir', help='The directory of the voc root.')
    parser.add_argument('--dataset', default=None, type=str,
                        dest='dataset', help='The target dataset that will be generated.')

    args = parser.parse_args()

    voc_det_generator = VocDetGenerator(args)
    voc_det_generator.generate_label()
