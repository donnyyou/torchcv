#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Parse json file of image labels.


import json
import cv2
import os
import argparse

from lib.tools.util.configer import Configer
from lib.tools.util.logger import Logger as Log


class ClsParser(object):
    def __init__(self, configer):
        self.configer = configer

    def parse_dir_cls(self, data_dir, json_file):
        if data_dir is None or not os.path.exists(data_dir):
            Log.error('Image Dir: {} not existed.'.format(data_dir))
            return

        if json_file is None or not os.path.exists(json_file):
            Log.error('Json Dir: {} not existed.'.format(json_file))
            return

        with open(json_file, 'r') as json_stream:
            info_tree = json.load(json_stream)
            for item_info in info_tree:
                image_file = os.path.join(data_dir, item_info['image_path'])
                image_canvas = cv2.imread(image_file)
                image_canvas = self.draw_label(image_canvas, item_info['label'])
                cv2.imshow('main', image_canvas)
                cv2.waitKey()

    def draw_label(self, image_canvas, label_id):
        class_name = self.configer.get('details', 'name_seq')[label_id]

        color_num = len(self.configer.get('details', 'color_list'))

        cv2.putText(image_canvas, class_name,
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=self.configer.get('details', 'color_list')[label_id % color_num], thickness=2)

        return image_canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None, type=str,
                        dest='config_file', help='The file of the hyper parameters.')
    parser.add_argument('--data_dir', default=None, type=str,
                        dest='data_dir', help='The data dir of Det Parser.')
    parser.add_argument('--json_file', default=None, type=str,
                        dest='json_file', help='The json file of Det Parser.')

    args_parser = parser.parse_args()

    det_parser = ClsParser(Configer(config_file=args_parser.config_file))
    det_parser.parse_dir_cls(args_parser.data_dir, args_parser.json_file)
