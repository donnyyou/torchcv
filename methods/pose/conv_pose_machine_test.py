#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (donnyyou@163.com)
# Test class for convolutional pose machine.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter

from datasets.pose_data_loader import PoseDataLoader
from datasets.tools.data_transformer import DataTransformer
from methods.tools.blob_helper import BlobHelper
from methods.tools.module_utilizer import ModuleUtilizer
from models.pose_model_manager import PoseModelManager
from utils.helpers.file_helper import FileHelper
from utils.helpers.image_helper import ImageHelper
from utils.layers.pose.heatmap_generator import HeatmapGenerator
from utils.tools.logger import Logger as Log
from vis.visualizer.pose_visualizer import PoseVisualizer


class ConvPoseMachineTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.pose_vis = PoseVisualizer(configer)
        self.pose_model_manager = PoseModelManager(configer)
        self.pose_data_loader = PoseDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.data_transformer = DataTransformer(configer)
        self.heatmap_generator = HeatmapGenerator(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.pose_net = None

        self._init_model()

    def _init_model(self):
        self.pose_net = self.pose_model_manager.multi_pose_detector()
        self.pose_net = self.module_utilizer.load_net(self.pose_net)
        self.pose_net.eval()

    def __test_img(self, image_path, save_path):
        Log.info('Image Path: {}'.format(image_path))
        ori_image = ImageHelper.read_image(image_path,
                                           tool=self.configer.get('data', 'image_tool'),
                                           mode=self.configer.get('data', 'input_mode'))

        ori_width, ori_height = ImageHelper.get_size(ori_image)
        ori_img_bgr = ImageHelper.get_cv2_bgr(ori_image, mode=self.configer.get('data', 'input_mode'))
        heatmap_avg = np.zeros((ori_height, ori_width, self.configer.get('network', 'heatmap_out')))
        for i, scale in enumerate(self.configer.get('test', 'scale_search')):
            image = self.blob_helper.make_input(ori_image,
                                                input_size=self.configer.get('test', 'input_size'),
                                                scale=scale)
            with torch.no_grad():
                heatmap_out_list = self.pose_net(image)
                heatmap_out = heatmap_out_list[-1]

                # extract outputs, resize, and remove padding
                heatmap = heatmap_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                heatmap = cv2.resize(heatmap, (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)

                heatmap_avg = heatmap_avg + heatmap / len(self.configer.get('test', 'scale_search'))

        all_peaks = self.__extract_heatmap_info(heatmap_avg)
        image_canvas = self.__draw_key_point(all_peaks, ori_img_bgr)
        ImageHelper.save(image_canvas, save_path)

    def __extract_heatmap_info(self, heatmap_avg):
        all_peaks = []

        for part in range(self.configer.get('network', 'heatmap_out') - 1):
            map_ori = heatmap_avg[:, :, part]
            map_gau = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map_gau.shape)
            map_left[1:, :] = map_gau[:-1, :]
            map_right = np.zeros(map_gau.shape)
            map_right[:-1, :] = map_gau[1:, :]
            map_up = np.zeros(map_gau.shape)
            map_up[:, 1:] = map_gau[:, :-1]
            map_down = np.zeros(map_gau.shape)
            map_down[:, :-1] = map_gau[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map_gau >= map_left, map_gau >= map_right, map_gau >= map_up,
                 map_gau >= map_down, map_gau > self.configer.get('vis', 'part_threshold')))

            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
            peaks = list(peaks)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

            all_peaks.append(peaks_with_score)

        return all_peaks

    def __draw_key_point(self, all_peaks, img_raw):
        img_canvas = img_raw.copy()  # B,G,R order

        for i in range(self.configer.get('network', 'heatmap_out') - 1):
            for j in range(len(all_peaks[i])):
                cv2.circle(img_canvas, all_peaks[i][j][0:2], self.configer.get('vis', 'stick_width'),
                           self.configer.get('details', 'color_list')[i], thickness=-1)

        return img_canvas

    def debug(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/pose', self.configer.get('dataset'), 'debug')

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for i, data_dict in enumerate(self.pose_data_loader.get_trainloader()):
            inputs = data_dict['img']
            input_size = [inputs.size(3), inputs.size(2)]
            heatmap = self.heatmap_generator(data_dict['kpts'], input_size)

            for j in range(inputs.size(0)):
                image_bgr = self.blob_helper.tensor2bgr(inputs[j])
                heatmap_avg = heatmap[j].numpy().transpose(1, 2, 0)
                heatmap_avg = cv2.resize(heatmap_avg, (0, 0), fx=self.configer.get('network', 'stride'),
                                     fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
                all_peaks = self.__extract_heatmap_info(heatmap_avg)
                image_save = self.__draw_key_point(all_peaks, image_bgr)
                cv2.imwrite(os.path.join(base_dir, '{}_{}_result.jpg'.format(i, j)), image_save)

    def test(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/pose', self.configer.get('dataset'))

        test_img = self.configer.get('test_img')
        test_dir = self.configer.get('test_dir')
        if test_img is None and test_dir is None:
            Log.error('test_img & test_dir not exists.')
            exit(1)

        if test_img is not None and test_dir is not None:
            Log.error('Either test_img or test_dir.')
            exit(1)

        if test_img is not None:
            base_dir = os.path.join(base_dir, 'test_img')
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            filename = test_img.rstrip().split('/')[-1]
            save_path = os.path.join(base_dir, filename)
            self.__test_img(test_img, save_path)

        else:
            base_dir = os.path.join(base_dir, 'test_dir',  test_dir.rstrip('/').split('/')[-1])
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            for filename in FileHelper.list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                save_path = os.path.join(base_dir, filename)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                self.__test_img(image_path, save_path)

