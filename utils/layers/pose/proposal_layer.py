#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Make proposals that each consists of all possible keypoints.


import math
import numpy as np
import torch
from scipy.spatial.distance import cosine
from scipy.ndimage.filters import gaussian_filter
from torch.autograd import Variable



class ProposalLayer(object):
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

        stride = self.configer.get('network', 'stride')
        peaks = list(zip(np.nonzero(peaks_binary)[1],
                         np.nonzero(peaks_binary)[0]))

        # A point format: (w, h, score, number)
        peaks_with_score = [[(x[0] + stride // 2) // stride,
                             (x[1] + stride // 2) // stride,
                             s_map[x[1], x[0]]] for x in peaks]

        return peaks_with_score

    def __get_simarray(self, peaks_vector):
        sim_array = np.zeros((len(peaks_vector), len(peaks_vector)))
        for i in range(len(peaks_vector)):
            for j in range(len(peaks_vector)):
                vec1 = peaks_vector[i]
                vec2 = peaks_vector[j]
                score = cosine(vec1, vec2)
                sim_array[i][j] = score

        return sim_array

    def __get_all_peaks(self, heatmap):
        all_peaks = []   # all of the possible points by classes.
        peak_counter = 0

        for part in range(self.configer.get('num_keypoints')):
            map_ori = heatmap[part, :, :]
            peaks_with_score = self.__get_peaks(map_ori)
            ids = range(peak_counter, peak_counter + len(peaks_with_score))
            peaks_with_score_and_id = [peaks_with_score[i] + [ids[i], ] for i in range(len(ids))]

            # print len(peaks_with_score_and_id)
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks_with_score_and_id)

        candidate = np.array([item for sublist in all_peaks for item in sublist])
        return all_peaks, candidate

    def __find_top_k_subsets(self, peak, subsets, sim_array):
        top_k_index = list()
        score_dict = dict()
        for i in range(len(subsets)):
            score_temp = 0.0
            points_count = 0
            for j in range(self.configer.get('num_keypoints')):
                if subsets[i][j] != -1:
                    score_temp += sim_array[i][peak[3]]
                    points_count += 1

            score_dict[i] = score_temp / (points_count * 1.0)

        for key, value in sorted(score_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True):
            top_k_index.append(key)

        return top_k_index[:self.configer.get('vis', 'top_k')]

    def __get_subsets(self, all_peaks, sim_array):
        subsets = -1 * np.ones((0, self.configer.get('num_keypoints')+2))
        for i in range(self.configer.get('num_keypoints')):
            subsets_new = -1 * np.ones((0, self.configer.get('num_keypoints')+2))
            update_index = list()
            for peak in all_peaks[i]:
                index_list = self.__find_top_k_subsets(peak, subsets, sim_array)
                if len(index_list) > 0:
                    for index in index_list:
                        subset = subsets[index].copy()
                        subset[i] = peak[3]
                        subsets_new = np.vstack([subsets_new, subset])
                        update_index.append(index)

                # Use score threshold to filter part ones.
                subset = -1 * np.ones(self.configer.get('num_keypoints')+2)
                subset[i] = peak[3]
                subsets_new = np.vstack([subsets_new, subset])

            subsets = np.delete(subsets, update_index, axis=0)
            subsets = np.vstack([subsets, subsets_new])

        return subsets

    def __cluster(self, inputs, mask):
        """
          Group into individuals.
        """
        proposals_list = list()
        candidates_list = list()

        for i in range(self.configer.get('batch_size')):
            heatmap = inputs[i, :, :, :].view(self.configer.get('num_keypoints'),
                                              self.configer.get('capsule', 'l_vec'),
                                              inputs.size(2), inputs.size(3))

            heatmap = torch.sqrt((heatmap * heatmap).sum(1)).squeeze()

            vecmap = inputs[i, :, :, :].view(self.configer.get('num_keypoints'),
                                             self.configer.get('capsule', 'l_vec'),
                                             inputs.size(2), inputs.size(3))

            all_peaks, candidate = self.__get_all_peaks(heatmap)

            peaks_vector = list()
            for j in range(self.configer.get('num_keypoints')):
                for peak in all_peaks[i]:
                    peaks_vector.append(vecmap[j, :, peak[0], peak[1]])

            sim_array = self.__get_simarray(peaks_vector)

            subsets = self.__get_subsets(all_peaks, sim_array)

            proposals_list.append(subsets)
            candidates_list.append(candidate)

        return proposals_list, candidates_list

    def __make_features_labels(self, inputs, proposals_list, candidates_list, kpts):
        # Find capsules for every proposals.
        # features = Variable(torch.zeros(len(proposals_list), self.l_vec), requires_grad=True).cuda()
        features_list = list()
        labels_list = list()
        proposal_count = 0

        for i in range(self.configer.get('batch_size')):
            candidate = candidates_list[i]
            proposals = proposals_list[i]
            vecmap = inputs[i, :, :, :].view(self.configer.get('num_keypoints'),
                                             self.configer.get('capsule', 'l_vec'),
                                             inputs.size(2), inputs.size(3))
            for proposal in proposals:
                feature = Variable(torch.zeros(self.configer.get('num_keypoints'),
                                               self.configer.get('capsule', 'l_vec')), require_grad=True).cuda()

                for j in range(self.configer.get('num_keypoints')):
                    vec = vecmap[j, :, candidate[proposal[i]][1], candidate[proposal[i]][0]]
                    feature[j, :] = vec

                features_list.append(feature)
                label = self.__get_label(proposal, candidates_list[i], kpts[i])
                labels_list.append(label)
                proposal_count += 1

        features = Variable(torch.zeros(proposal_count, self.configer.get('num_keypoints'),
                                        self.configer.get('capsule', 'l_vec')), requires_grad=True).cuda()

        labels = Variable(torch.zeros(proposal_count,
                                      self.configer.get('num_keypoints')), requires_grad=True).cuda()

        for i in range(proposal_count):
            features[i, :, :] = features_list[i][:, :]
            labels[i, :] = labels_list[i][:, :]

        return features, labels

    def __get_label(self, proposal, kpts, candidates):
        label = Variable(torch.zeros(self.configer.get('num_keypoints'),), require_grad=True).cuda()

        stride = 8.0
        max_score = 0.0
        start = stride / 2.0 - 0.5
        sigma = self.configer.get('heatmap', 'sigma')

        for i in range(len(kpts)):
            score_temp = 0.0
            visiable_count = 0
            label_temp = Variable(torch.zeros(self.configer.get('num_keypoints'),), require_grad=True).cuda()
            for j in range(self.configer.get('num_keypoints')):
                if kpts[i][j][2] > 1:
                    continue

                visiable_count += 1
                x = kpts[i][j][0]
                y = kpts[i][j][1]
                xx = candidates[proposal[j]][1]
                yy = candidates[proposal[j]][0]
                xx = start + xx * stride
                yy = start + yy * stride
                dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                if dis > 4.6052:
                    continue

                score_temp += math.exp(-dis)
                label_temp[j] = math.exp(-dis)

            if score_temp/(visiable_count*1.0) > max_score:
                max_score = score_temp
                label = label_temp

        return label

    def forward(self, inputs, kpts, mask, is_label=True):
        proposals_list, candidates_list = self.__cluster(inputs, mask)
        features, labels = self.__make_features_labels(proposals_list, candidates_list, kpts)

        if is_label:
            return features, labels
        else:
            return features, proposals_list


if __name__ == "__main__":
    # Test proposal layer.
    pass
