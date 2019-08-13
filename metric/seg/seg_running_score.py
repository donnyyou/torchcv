#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Segmentation running score.


import numpy as np


class SegRunningScore(object):

    def __init__(self, configer):
        self.configer = configer
        self.n_classes = self.configer.get('data', 'num_classes')
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)

        return hist

    def update(self, label_preds, label_trues):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def _get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return acc, acc_cls, fwavacc, mean_iu, cls_iu

    def get_mean_iou(self):
        return self._get_scores()[3]

    def get_cls_iou(self):
        return self._get_scores()[4]

    def get_pixel_acc(self):
        return self._get_scores()[0]

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

