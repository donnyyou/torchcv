#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Parse json file of keypoints.


import json
import cv2
import os
import math
import numpy as np
import argparse

from lib.tools.util.configer import Configer
from lib.tools.util.logger import Logger as Log


class PoseParser(object):
    def __init__(self, configer):
        self.configer = configer

    def parse_img_pose(self, image_file, json_file, mask_file=None):
        if image_file is None or not os.path.exists(image_file):
            Log.error('Image file: {} not existed.'.format(image_file))
            return

        if json_file is None or not os.path.exists(json_file):
            Log.error('Json file: {} not existed.'.format(json_file))
            return

        image_canvas = cv2.imread(image_file)  # B, G, R order.

        with open(json_file, 'r') as json_stream:
            info_tree = json.load(json_stream)
            image_canvas = self.draw_points(image_canvas, info_tree)
            if self.configer.exists('details', 'limb_seq'):
                image_canvas = self.link_points(image_canvas, info_tree)

        if mask_file is not None:
            mask_canvas = cv2.imread(mask_file)
            image_canvas = cv2.addWeighted(image_canvas, 0.6, mask_canvas, 0.4, 0)

        cv2.imshow('main', image_canvas)
        cv2.waitKey()

    def parse_dir_pose(self, image_dir, json_dir, mask_dir=None):
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
                image_canvas = self.draw_points(image_canvas, info_tree)
                if self.configer.exists('details', 'limb_seq'):
                    image_canvas = self.link_points(image_canvas, info_tree)

            if mask_dir is not None:
                mask_file = os.path.join(mask_dir, '{}_vis.png'.format(shotname))
                mask_canvas = cv2.imread(mask_file)
                image_canvas = cv2.addWeighted(image_canvas, 0.6, mask_canvas, 0.4, 0)

            cv2.imshow('main', image_canvas)
            cv2.waitKey()

    def draw_points(self, image_canvas, info_tree):
        for object in info_tree['objects']:
            for i in range(self.configer.get('data', 'num_kpts')):
                if object['kpts'][i][2] == -1:
                    continue

                cv2.circle(image_canvas, (int(object['kpts'][i][0]), int(object['kpts'][i][1])),
                           self.configer.get('res', 'vis_circle_radius'),
                           self.configer.get('details', 'color_list')[i], thickness=-1)

        return image_canvas

    def link_points(self, image_canvas, info_tree):
        for object in info_tree['objects']:
            for i in range(len(self.configer.get('details', 'limb_seq'))):
                two_points = np.array(object['kpts'])[np.array(self.configer.get('details', 'limb_seq')[i])-1]
                if -1 in two_points[:, 2]:
                    continue

                cur_canvas = image_canvas.copy()

                X = two_points[:, 0]
                Y = two_points[:, 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                polygon = cv2.ellipse2Poly((int(mX), int(mY)),
                                           (int(length / 2),
                                            self.configer.get('res', 'vis_stick_width')), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, self.configer.get('details', 'color_list')[i])
                image_canvas = cv2.addWeighted(image_canvas, 0.4, cur_canvas, 0.6, 0)

        return image_canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None, type=str,
                        dest='config_file', help='The file of the hyper parameters.')
    parser.add_argument('--image_file', default=None, type=str,
                        dest='image_file', help='The image file of Pose Parser.')
    parser.add_argument('--json_file', default=None, type=str,
                        dest='json_file', help='The json file of Pose Parser.')
    parser.add_argument('--mask_file', default=None, type=str,
                        dest='mask_file', help='The mask file of Pose Parser.')
    parser.add_argument('--image_dir', default=None, type=str,
                        dest='image_dir', help='The image directory of Pose Parser.')
    parser.add_argument('--json_dir', default=None, type=str,
                        dest='json_dir', help='The json directory of Pose Parser.')
    parser.add_argument('--mask_dir', default=None, type=str,
                        dest='mask_dir', help='The mask directory of Pose Parser.')

    args_parser = parser.parse_args()

    pose_parser = PoseParser(Configer(config_file=args_parser.config_file))
    pose_parser.parse_img_pose(args_parser.image_file, args_parser.json_file, args_parser.mask_file)
    pose_parser.parse_dir_pose(args_parser.image_dir, args_parser.json_dir, args_parser.mask_dir)
