#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Parse label file of segmentation.


import cv2
import os
import numpy as np
import argparse
from PIL import Image

from lib.tools.util.configer import Configer
from lib.tools.util.logger import Logger as Log


class SegParser(object):
    def __init__(self, configer):
        self.configer = configer

    def parse_img_seg(self, image_file, label_file):
        if image_file is None or not os.path.exists(image_file):
            Log.error('Image file: {} not existed.'.format(image_file))
            return

        if label_file is None or not os.path.exists(label_file):
            Log.error('Label file: {} not existed.'.format(label_file))
            return

        image_canvas = cv2.imread(image_file)  # B, G, R order.

        mask_canvas = self.colorize(np.array(Image.open(label_file).convert('P')))
        image_canvas = cv2.addWeighted(image_canvas, 0.6, mask_canvas, 0.4, 0)

        cv2.imshow('main', image_canvas)
        cv2.waitKey()

    def parse_dir_seg(self, image_dir, label_dir):
        if image_dir is None or not os.path.exists(image_dir):
            Log.error('Image Dir: {} not existed.'.format(image_dir))
            return

        if label_dir is None or not os.path.exists(label_dir):
            Log.error('Label Dir: {} not existed.'.format(label_dir))
            return

        for image_file in os.listdir(image_dir):
            shotname, extension = os.path.splitext(image_file)
            Log.info(image_file)
            image_canvas = cv2.imread(os.path.join(image_dir, image_file))  # B, G, R order.
            label_file = os.path.join(label_dir, '{}.png'.format(shotname))
            mask_canvas = self.colorize(np.array(Image.open(label_file).convert('P')))
            image_canvas = cv2.addWeighted(image_canvas, 0.6, mask_canvas, 0.4, 0)

            cv2.imshow('main', image_canvas)
            cv2.waitKey()

    def colorize(self, label_map, image_canvas=None):
        height, width = label_map.shape
        color_dst = np.zeros((height, width, 3), dtype=np.uint8)
        color_list = self.configer.get('details', 'color_list')
        for i in range(self.configer.get('data', 'num_classes')):
            color_dst[label_map == i] = color_list[i % len(color_list)]

        color_img_rgb = np.array(color_dst, dtype=np.uint8)
        color_img_bgr = cv2.cvtColor(color_img_rgb, cv2.COLOR_RGB2BGR)

        if image_canvas is not None:
            image_canvas = cv2.addWeighted(image_canvas, 0.6, color_img_bgr, 0.4, 0)
            return image_canvas

        else:
            return color_img_bgr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None, type=str,
                        dest='config_file', help='The file of the hyper parameters.')
    parser.add_argument('--image_file', default=None, type=str,
                        dest='image_file', help='The image file of Seg Parser.')
    parser.add_argument('--label_file', default=None, type=str,
                        dest='label_file', help='The label file of Seg Parser.')
    parser.add_argument('--image_dir', default=None, type=str,
                        dest='image_dir', help='The image directory of Seg Parser.')
    parser.add_argument('--label_dir', default=None, type=str,
                        dest='label_dir', help='The label directory of Seg Parser.')

    args_parser = parser.parse_args()

    seg_parser = SegParser(Configer(config_file=args_parser.config_file))
    seg_parser.parse_img_seg(args_parser.image_file, args_parser.label_file)
    seg_parser.parse_dir_seg(args_parser.image_dir, args_parser.label_dir)
