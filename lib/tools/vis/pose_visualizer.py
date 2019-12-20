#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualizer for pose estimator.


import os

import cv2
import matplotlib
import numpy as np
import pylab as plt
from PIL import Image
from numpy import ma
from scipy.ndimage.filters import gaussian_filter

from lib.data.transforms import DeNormalize
from lib.tools.util.logger import Logger as Log


POSE_DIR = 'vis/results/pose'


class PoseVisualizer(object):

    def __init__(self, configer):
        self.configer = configer

    def __get_peaks(self, heatmap):
        s_map = gaussian_filter(heatmap, sigma=3)
        map_left = np.zeros(s_map.shape)
        map_left[:, 1:] = s_map[:, :-1]
        map_right = np.zeros(s_map.shape)
        map_right[:, :-1] = s_map[:, 1:]
        map_up = np.zeros(s_map.shape)
        map_up[1:, :] = s_map[:-1, :]
        map_down = np.zeros(s_map.shape)
        map_down[:-1, :] = s_map[1:, :]

        # Get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
            (s_map >= map_left, s_map >= map_right,
             s_map >= map_up, s_map >= map_down,
             s_map > self.configer.get('vis', 'part_threshold')))

        peaks = list(zip(np.nonzero(peaks_binary)[1],
                         np.nonzero(peaks_binary)[0]))

        # A point format: (w, h, score, number)
        peaks_with_score = [x + (s_map[x[1], x[0]],) for x in peaks]
        return peaks_with_score

    def vis_peaks(self, heatmap_in, ori_img_in, name='default', sub_dir='peaks'):
        base_dir = os.path.join(self.configer.get('project_dir'), POSE_DIR, sub_dir)
        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        if not isinstance(heatmap_in, np.ndarray):
            if len(heatmap_in.size()) != 3:
                Log.error('Heatmap size is not valid.')
                exit(1)

            heatmap = heatmap_in.clone().data.cpu().numpy().transpose(1, 2, 0)
        else:
            heatmap = heatmap_in.copy()

        if not isinstance(ori_img_in, np.ndarray):
            ori_img = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                  mean=self.configer.get('normalize', 'mean'),
                                  std=self.configer.get('normalize', 'std'))(ori_img_in.clone())
            ori_img = ori_img.data.cpu().squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        else:
            ori_img = ori_img_in.copy()

        for j in range(self.configer.get('data', 'num_kpts')):
            peaks = self.__get_peaks(heatmap[:, :, j])

            for peak in peaks:
                ori_img = cv2.circle(ori_img, (peak[0], peak[1]),
                                     self.configer.get('vis', 'circle_radius'),
                                     self.configer.get('details', 'color_list')[j], thickness=-1)

            cv2.imwrite(os.path.join(base_dir, '{}_{}.jpg'.format(name, j)), ori_img)

    def vis_paf(self, inputs_in, ori_img_in, name='default', sub_dir='pafs'):
        base_dir = os.path.join(self.configer.get('project_dir'), POSE_DIR, sub_dir)
        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        if not isinstance(inputs_in, np.ndarray):
            if len(inputs_in.size()) != 3:
                Log.error('Pafs size is not valid.')
                exit(1)

            inputs = inputs_in.clone().data.squeeze().cpu().numpy().transpose(1, 2, 0)
        else:
            inputs = inputs_in.copy()

        if not isinstance(ori_img_in, np.ndarray):
            if len(ori_img_in.size()) != 3:
                Log.error('Image size is not valid.')
                exit(1)

            ori_img = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                  mean=self.configer.get('normalize', 'mean'),
                                  std=self.configer.get('normalize', 'std'))(ori_img_in.clone())
            ori_img = ori_img.data.cpu().squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)
        else:
            ori_img = ori_img_in.copy()

        for i in range(len(self.configer.get('details', 'limb_seq'))):
            U = inputs[:, :, 2*i] * -1
            V = inputs[:, :, 2*i+1]
            X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
            M = np.zeros(U.shape, dtype='bool')
            M[U ** 2 + V ** 2 < 0.5 * 0.5] = True
            U = ma.masked_array(U, mask=M)
            V = ma.masked_array(V, mask=M)

            # 1
            img = Image.fromarray(ori_img, mode='RGB')
            plt.figure()
            plt.imshow(img, alpha=1.0)
            s = 5
            Q = plt.quiver(X[::s, ::s], Y[::s, ::s], U[::s, ::s], V[::s, ::s],
                           scale=50, headaxislength=4, alpha=1.0, width=0.001, color='r')

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(10, 10)
            plt.savefig(os.path.join(base_dir, '{}_{}.jpg'.format(name, i)))


if __name__ == "__main__":
    # Test the visualizer.
    pass
