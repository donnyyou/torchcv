#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Parse json file of bounding boxes.


import json
import cv2
import os
import argparse

from lib.tools.util.configer import Configer
from lib.tools.util.logger import Logger as Log


class DetParser(object):
    def __init__(self, configer):
        self.configer = configer

    def parse_img_det(self, image_file, json_file):
        if image_file is None or not os.path.exists(image_file):
            Log.error('Image file: {} not existed.'.format(image_file))
            return

        if json_file is None or not os.path.exists(json_file):
            Log.error('Json file: {} not existed.'.format(json_file))
            return

        image_canvas = cv2.imread(image_file)  # B, G, R order.

        with open(json_file, 'r') as json_stream:
            info_tree = json.load(json_stream)
            image_canvas = self.draw_bboxes(image_canvas, info_tree)

        cv2.imshow('main', image_canvas)
        cv2.waitKey()

    def parse_dir_det(self, image_dir, json_dir):
        if image_dir is None or not os.path.exists(image_dir):
            Log.error('Image Dir: {} not existed.'.format(image_dir))
            return

        if json_dir is None or not os.path.exists(json_dir):
            Log.error('Json Dir: {} not existed.'.format(json_dir))
            return

        for image_file in os.listdir(image_dir):
            shotname, extension = os.path.splitext(image_file)
            Log.info(image_file)
            image_canvas = cv2.imread(os.path.join(image_dir, image_file))  # B, G, R order.
            with open(os.path.join(json_dir, '{}.json'.format(shotname)), 'r') as json_stream:
                info_tree = json.load(json_stream)
                image_canvas = self.draw_bboxes(image_canvas, info_tree)

            cv2.namedWindow("main", cv2.WINDOW_NORMAL)
            cv2.imshow('main', image_canvas)
            cv2.waitKey()

    def draw_bboxes(self, image_canvas, info_tree, conf_threshold=None):
        for object in info_tree['objects']:
            class_name = self.configer.get('details', 'name_seq')[object['label']]
            if 'score' in object:
                if conf_threshold is not None and float(object['score']) < conf_threshold:
                    continue

                class_name = '{}_{}'.format(class_name, object['score'])

            color_num = len(self.configer.get('details', 'color_list'))
            cv2.rectangle(image_canvas,
                          (int(object['bbox'][0]), int(object['bbox'][1])),
                          (int(object['bbox'][2]), int(object['bbox'][3])),
                          color=self.configer.get('details', 'color_list')[object['label'] % color_num], thickness=3)

            cv2.putText(image_canvas, class_name,
                        (int(object['bbox'][0]) + 5, int(object['bbox'][3]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=self.configer.get('details', 'color_list')[object['label'] % color_num], thickness=2)

        return image_canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='../../configs/det/coco/yolov3_darknet_coco_det.conf', type=str,
                        dest='config_file', help='The file of the hyper parameters.')
    parser.add_argument('--image_file', default=None, type=str,
                        dest='image_file', help='The image file of Det Parser.')
    parser.add_argument('--json_file', default=None, type=str,
                        dest='json_file', help='The json file of Det Parser.')
    parser.add_argument('--image_dir', default='/home/donny/DataSets/Pallet/train/image', type=str,
                        dest='image_dir', help='The image directory of Det Parser.')
    parser.add_argument('--json_dir', default='/home/donny/DataSets/Pallet/train/json', type=str,
                        dest='json_dir', help='The json directory of Det Parser.')

    args_parser = parser.parse_args()

    det_parser = DetParser(Configer(config_file=args_parser.config_file))
    det_parser.parse_img_det(args_parser.image_file, args_parser.json_file)
    det_parser.parse_dir_det(args_parser.image_dir, args_parser.json_dir)
