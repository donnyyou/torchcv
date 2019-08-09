#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Pose Estimation running score.


import numpy as np


class PoseRunningScore(object):
    def __init__(self, configer):
        self.configer = configer
        self.oks_all = np.zeros(0)
        self.oks_num = 0

    def compute_oks(self, gt_kpts, pred_kpts):
        """Compute oks matrix (size gtN*pN)."""
        gt_count = len(gt_kpts)
        pred_count = len(pred_kpts)
        oks = np.zeros((gt_count, pred_count))
        if pred_count == 0:
            return oks.T

        # for every human keypoint annotation
        for i in range(gt_count):
            anno_keypoints = np.reshape(np.array(gt_kpts[i]), (self.configer.get('data', 'num_keypoints'), 3))
            visible = anno_keypoints[:, 2] == 1
            scale = max(np.max(anno_keypoints[:, 0]) - np.min(anno_keypoints[:, 0]),
                        np.max(anno_keypoints[:, 0]) - np.min(anno_keypoints[:, 0])) ** 2 + 1e-8

            if np.sum(visible) == 0:
                for j in range(pred_count):
                    oks[i, j] = 0
            else:
                # for every predicted human
                for j in range(pred_count):
                    predict_keypoints = np.reshape(np.array(pred_kpts[j]),
                                                   (self.configer.get('data', 'num_keypoints'), 3))
                    dis = np.sum((anno_keypoints[visible, :2] - predict_keypoints[visible, :2]) ** 2, axis=1)
                    oks[i, j] = np.mean(
                        np.exp(-dis / 2 / self.configer.get('details', 'delta')[visible] ** 2 / (scale + 1))
                    )

        return oks

    def update(self, batch_pred_kpts, batch_gt_kpts):
        """Evaluate predicted_file and return mAP."""
        # Construct set to speed up id searching.
        # for every annotation in our test/validation set
        for i in range(len(batch_pred_kpts)):
            # if the image in the predictions, then compute oks
            oks = self.compute_oks(batch_gt_kpts[i], batch_pred_kpts[i])
            # view pairs with max OKSs as match ones, add to oks_all
            self.oks_all = np.concatenate((self.oks_all, np.max(oks, axis=1)), axis=0)
            # accumulate total num by max(gtN,pN)
            self.oks_num += np.max(oks.shape)

    def get_mAP(self):
        # compute mAP by APs under different oks thresholds
        average_precision = []
        for threshold in np.linspace(0.5, 0.95, 10):
            average_precision.append(np.sum(self.oks_all > threshold) / np.float32(self.oks_num))

        return np.mean(average_precision)

    def reset(self):
        self.oks_all = np.zeros(0)
        self.oks_num = 0

