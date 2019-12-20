#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Pose Estimator.


import math
import os
import cv2
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter

from data.pose.data_loader import DataLoader
from lib.runner.blob_helper import BlobHelper
from lib.runner.runner_helper import RunnerHelper
from model.pose.model_manager import ModelManager
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.helper.json_helper import JsonHelper
from lib.tools.util.logger import Logger as Log
from lib.tools.parser.pose_parser import PoseParser
from lib.tools.vis.pose_visualizer import PoseVisualizer


class OpenPoseTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.pose_visualizer = PoseVisualizer(configer)
        self.pose_parser = PoseParser(configer)
        self.pose_model_manager = ModelManager(configer)
        self.pose_data_loader = DataLoader(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.pose_net = None

        self._init_model()

    def _init_model(self):
        self.pose_net = self.pose_model_manager.get_pose_model()
        self.pose_net = RunnerHelper.load_net(self, self.pose_net)
        self.pose_net.eval()

    def _get_blob(self, ori_image, scale=None):
        assert scale is not None
        image = self.blob_helper.make_input(image=ori_image, scale=scale)

        b, c, h, w = image.size()
        border_hw = [h, w]
        if self.configer.exists('test', 'fit_stride'):
            stride = self.configer.get('test', 'fit_stride')

            pad_w = 0 if (w % stride == 0) else stride - (w % stride)  # right
            pad_h = 0 if (h % stride == 0) else stride - (h % stride)  # down

            expand_image = torch.zeros((b, c, h + pad_h, w + pad_w)).to(image.device)
            expand_image[:, :, 0:h, 0:w] = image
            image = expand_image

        return image, border_hw

    def __test_img(self, image_path, json_path, raw_path, vis_path):

        Log.info('Image Path: {}'.format(image_path))
        ori_image = ImageHelper.read_image(image_path,
                                           tool=self.configer.get('data', 'image_tool'),
                                           mode=self.configer.get('data', 'input_mode'))

        ori_width, ori_height = ImageHelper.get_size(ori_image)
        ori_img_bgr = ImageHelper.get_cv2_bgr(ori_image, mode=self.configer.get('data', 'input_mode'))
        heatmap_avg = np.zeros((ori_height, ori_width, self.configer.get('network', 'heatmap_out')))
        paf_avg = np.zeros((ori_height, ori_width, self.configer.get('network', 'paf_out')))
        multiplier = [scale * self.configer.get('test', 'input_size')[1] / ori_height
                      for scale in self.configer.get('test', 'scale_search')]
        stride = self.configer.get('network', 'stride')
        for i, scale in enumerate(multiplier):
            image, border_hw = self._get_blob(ori_image, scale=scale)
            with torch.no_grad():
                paf_out_list, heatmap_out_list = self.pose_net(image)
                paf_out = paf_out_list[-1]
                heatmap_out = heatmap_out_list[-1]

                # extract outputs, resize, and remove padding
                heatmap = heatmap_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                heatmap = cv2.resize(heatmap, None, fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
                heatmap = cv2.resize(heatmap[:border_hw[0], :border_hw[1]],
                                     (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)

                paf = paf_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                paf = cv2.resize(paf, None, fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
                paf = cv2.resize(paf[:border_hw[0], :border_hw[1]],
                                 (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)

                heatmap_avg = heatmap_avg + heatmap / len(multiplier)
                paf_avg = paf_avg + paf / len(multiplier)

        all_peaks = self.__extract_heatmap_info(heatmap_avg)
        special_k, connection_all = self.__extract_paf_info(ori_img_bgr, paf_avg, all_peaks)
        subset, candidate = self.__get_subsets(connection_all, special_k, all_peaks)
        json_dict = self.__get_info_tree(ori_img_bgr, subset, candidate)

        image_canvas = self.pose_parser.draw_points(ori_img_bgr.copy(), json_dict)
        image_canvas = self.pose_parser.link_points(image_canvas, json_dict)

        ImageHelper.save(image_canvas, vis_path)
        ImageHelper.save(ori_img_bgr, raw_path)
        Log.info('Json Save Path: {}'.format(json_path))
        JsonHelper.save_file(json_dict, json_path)

    def __get_info_tree(self, image_raw, subset, candidate):
        json_dict = dict()
        height, width, _ = image_raw.shape
        json_dict['image_height'] = height
        json_dict['image_width'] = width
        object_list = list()
        for n in range(len(subset)):
            if subset[n][-1] < self.configer.get('res', 'num_threshold'):
                continue

            if subset[n][-2] / subset[n][-1] < self.configer.get('res', 'avg_threshold'):
                continue

            object_dict = dict()
            object_dict['kpts'] = np.zeros((self.configer.get('data', 'num_kpts'), 3)).tolist()
            for j in range(self.configer.get('data', 'num_kpts')):
                index = subset[n][j]
                if index == -1:
                    object_dict['kpts'][j][0] = -1
                    object_dict['kpts'][j][1] = -1
                    object_dict['kpts'][j][2] = -1

                else:
                    object_dict['kpts'][j][0] = candidate[index.astype(int)][0]
                    object_dict['kpts'][j][1] = candidate[index.astype(int)][1]
                    object_dict['kpts'][j][2] = 1

            object_dict['score'] = subset[n][-2]
            object_list.append(object_dict)

        json_dict['objects'] = object_list
        return json_dict

    def __extract_heatmap_info(self, heatmap_avg):
        all_peaks = []
        peak_counter = 0

        for part in range(self.configer.get('data', 'num_kpts')):
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
                 map_gau >= map_down, map_gau > self.configer.get('res', 'part_threshold')))

            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
            peaks = list(peaks)

            '''
            del_flag = [0 for i in range(len(peaks))]
            for i in range(len(peaks)):
                if del_flag[i] == 0:
                    for j in range(i+1, len(peaks)):
                        if max(abs(peaks[i][0] - peaks[j][0]), abs(peaks[i][1] - peaks[j][1])) <= 6:
                            del_flag[j] = 1

            new_peaks = list()
            for i in range(len(peaks)):
                if del_flag[i] == 0:
                    new_peaks.append(peaks[i])

            peaks = new_peaks
            '''

            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            ids = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (ids[i],) for i in range(len(ids))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        return all_peaks

    def __extract_paf_info(self, img_raw, paf_avg, all_peaks):
        connection_all = []
        special_k = []
        mid_num = self.configer.get('res', 'mid_point_num')

        for k in range(len(self.configer.get('details', 'limb_seq'))):
            score_mid = paf_avg[:, :, [k*2, k*2+1]]
            candA = all_peaks[self.configer.get('details', 'limb_seq')[k][0] - 1]
            candB = all_peaks[self.configer.get('details', 'limb_seq')[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-9
                        vec = np.divide(vec, norm)

                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num))
                        startend = list(startend)

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts)
                        score_with_dist_prior += min(0.5 * img_raw.shape[0] / norm - 1, 0)

                        num_positive = len(np.nonzero(score_midpts > self.configer.get('res', 'limb_threshold'))[0])
                        criterion1 = num_positive > int(self.configer.get('res', 'limb_pos_ratio') * len(score_midpts))
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        return special_k, connection_all

    def __get_subsets(self, connection_all, special_k, all_peaks):
        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, self.configer.get('data', 'num_kpts') + 2))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in self.configer.get('details', 'mini_tree'):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(self.configer.get('details', 'limb_seq')[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found:
                        row = -1 * np.ones(self.configer.get('data', 'num_kpts') + 2)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        return subset, candidate

    def debug(self, vis_dir):
        for i, data_dict in enumerate(self.pose_data_loader.get_trainloader()):
            inputs = data_dict['img']
            maskmap = data_dict['maskmap']
            heatmap = data_dict['heatmap']
            vecmap = data_dict['vecmap']
            for j in range(inputs.size(0)):
                count = count + 1
                if count > 10:
                    exit(1)

                Log.info(heatmap.size())
                image_bgr = self.blob_helper.tensor2bgr(inputs[j])
                mask_canvas = maskmap[j].repeat(3, 1, 1).numpy().transpose(1, 2, 0)
                mask_canvas = (mask_canvas * 255).astype(np.uint8)
                mask_canvas = cv2.resize(mask_canvas, (0, 0), fx=self.configer.get('network', 'stride'),
                                         fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)

                image_bgr = cv2.addWeighted(image_bgr, 0.6, mask_canvas, 0.4, 0)
                heatmap_avg = heatmap[j].numpy().transpose(1, 2, 0)
                heatmap_avg = cv2.resize(heatmap_avg, (0, 0), fx=self.configer.get('network', 'stride'),
                                     fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
                paf_avg = vecmap[j].numpy().transpose(1, 2, 0)
                paf_avg = cv2.resize(paf_avg, (0, 0), fx=self.configer.get('network', 'stride'),
                                     fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
                self.pose_visualizer.vis_peaks(heatmap_avg, image_bgr)
                self.pose_visualizer.vis_paf(paf_avg, image_bgr)
                all_peaks = self.__extract_heatmap_info(heatmap_avg)
                special_k, connection_all = self.__extract_paf_info(image_bgr, paf_avg, all_peaks)
                subset, candidate = self.__get_subsets(connection_all, special_k, all_peaks)
                json_dict = self.__get_info_tree(image_bgr, subset, candidate)
                image_canvas = self.pose_parser.draw_points(image_bgr, json_dict)
                image_canvas = self.pose_parser.link_points(image_canvas, json_dict)
                cv2.imwrite(os.path.join(vis_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()

