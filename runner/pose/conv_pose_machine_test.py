#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


import os
import cv2
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter

from lib.runner.blob_helper import BlobHelper
from lib.runner.runner_helper import RunnerHelper
from model.pose.model_manager import ModelManager
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as Log


class ConvPoseMachineTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.pose_model_manager = ModelManager(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.pose_net = None

        self._init_model()

    def _init_model(self):
        self.pose_net = self.pose_model_manager.get_pose_model()
        self.pose_net = RunnerHelper.load_net(self, self.pose_net)
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

