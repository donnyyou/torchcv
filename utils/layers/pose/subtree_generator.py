#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Make proposals that each consists of all possible keypoints.


import os
import math
import time
import numpy as np
import sys
import torch
import cv2
from scipy.spatial.distance import cosine
from scipy.ndimage.filters import gaussian_filter


class SubtreeGenerator(object):
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

    def __get_connections(self, all_peaks, vecmap):
        connection_all = list()
        special_k = list()

        for k in range(len(self.configer.get('coco', 'limb_seq'))):
            canda = all_peaks[self.configer.get('coco', 'limb_seq')[k][0]-1]
            candb = all_peaks[self.configer.get('coco', 'limb_seq')[k][1]-1]
            lena = len(canda)
            lenb = len(candb)
            print "%d %d\n" % (lena, lenb)

            if lena != 0 and lenb != 0:
                connection_candidate = []
                for i in range(lena):
                    for j in range(lenb):
                        vec1 = vecmap[self.configer.get('coco', 'limb_seq')[k][0], canda[i][1], canda[i][0]]
                        vec2 = vecmap[self.configer.get('coco', 'limb_seq')[k][1], candb[j][1], candb[j][0]]
                        score_with_dist_prior = 1.0 - np.sqrt(((vec1 - vec2)*(vec1 - vec2)).sum())
                        print score_with_dist_prior

                        if score_with_dist_prior > self.configer.get('vis', 'limb_threshold'):
                            connection_candidate.append([i, j,
                                                         score_with_dist_prior,
                                                         score_with_dist_prior + canda[i][2] + candb[j][2]])

                # sort the possible line from large to small order.
                connection_candidate = sorted(connection_candidate, key=lambda x: x[3], reverse=True)
                connection = np.zeros((0, 5))

                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0: 3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        # the number of a point, the number of b point, score, a point, b point
                        connection = np.vstack([connection, [canda[i][3], candb[j][3], s, i, j]])
                        if len(connection) >= min(lena, lenb):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        return connection_all, special_k

    def __get_proposals(self, connection_all, candidate, special_k):
        subset = -1 * np.ones((0, 20))
        for k in range(len(self.configer.get('coco', 'limb_seq'))):
            if k not in special_k:
                partas = connection_all[k][:, 0]
                partbs = connection_all[k][:, 1]
                indexa, indexb = np.array(self.configer.get('coco', 'limb_seq')[k]) - 1

                for i in range(len(connection_all[k])):
                    found = 0
                    flag = [False, False]
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        # fix the bug, found == 2 and not joint will lead someone occur more than once.
                        # if more than one, we choose the subset, which has a higher score.
                        if subset[j][indexa] == partas[i]:
                            if flag[0] == False:
                                flag[0] = found
                                subset_idx[found] = j
                                flag[0] = True
                                found += 1
                            else:
                                ids = subset_idx[flag[0]]
                                if subset[ids][-1] < subset[j][-1]:
                                    subset_idx[flag[0]] = j
                        if subset[j][indexb] == partbs[i]:
                            if flag[1] == False:
                                flag[1] = found
                                subset_idx[found] = j
                                flag[1] = True
                                found += 1
                            else:
                                ids = subset_idx[flag[1]]
                                if subset[ids][-1] < subset[j][-1]:
                                    subset_idx[flag[1]] = j

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexb] != partbs[i]):
                            subset[j][indexb] = partbs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partbs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found equals to 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexb] = partbs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partbs[i].astype(int), 2] + connection_all[k][i][2]
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexa] = partas[i]
                        row[indexb] = partbs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) \
                                  + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete som rows of subset which has few parts occur
        deleteidx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteidx.append(i)

        subset = np.delete(subset, deleteidx, axis=0)
        return subset

    def __get_all_peaks(self, heatmap, vecmap, mask):
        all_peaks = []   # all of the possible points by classes.
        peak_counter = 0

        for part in range(self.configer.get('num_keypoints')):
            map_ori = heatmap[:, :, part]
            peaks_with_score = self.__get_peaks(map_ori)
            ids = range(peak_counter, peak_counter + len(peaks_with_score))
            peaks_with_score_and_id = [peaks_with_score[i] + (ids[i], ) for i in range(len(ids))]

            # print len(peaks_with_score_and_id)
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(len(peaks_with_score_and_id))

        candidate = np.array([item for sublist in all_peaks for item in sublist])
        return all_peaks, candidate

    def __search_subtree(self, inputs, mask):
        heatmap = inputs.view(-1, self.l_vec, inputs.size(2), inputs.size(3))
        heatmap = torch.sqrt((heatmap*heatmap).sum(1)).squeeze()
        vecmap = inputs.view(-1, self.l_vec, inputs.size(2), inputs.size(3))

        all_peaks, candidate = self.__get_all_peaks(heatmap, mask)

        connection_all = [] # save all of the possible lines by classes.
        special_k = []      # save the lines, which haven't legal points.

        connection_all, special_k = self.__get_connections(all_peaks, vecmap)

        subset = self.__get_proposals(connection_all, candidate, special_k)
        return subset


    def __make_features(self, proposals, inputs):
        # find capsules for every proposals.
        pass

    def __make_labels(self, proposals, kpts):
        pass

    def __make_proposals(self, inputs, mask=None):
        proposals_list = list()
        for i in range(self.configer.get('batch_size')):
            proposals = self.__search_subtree(inputs[i, :, :, :])
            proposals_list.append(proposals)

        return proposals_list

    def generate_subtree(self, inputs, kpts, is_label=True):
        proposals = self.__make_proposals(inputs)
        features = self.__make_features(proposals, inputs)
        labels = self.__make_labels(proposals, kpts)
        if is_label:
            return features, labels
        else:
            return features, proposals


if __name__ == "__main__":
    # Test proposal layer.
    pass
