#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import os
import numpy as np
import argparse

from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as Log
from lib.tools.util.configer import Configer
from metric.seg.seg_running_score import SegRunningScore


class SegEvaluator(object):
    def __init__(self, configer):
        self.configer = configer
        self.seg_running_score = SegRunningScore(configer)

    def relabel(self, labelmap):
        if self.configer.get('data.reduce_zero_label', default=False):
            labelmap[labelmap == 0] = 255
            labelmap = (labelmap - 1).astype(np.uint8)
            labelmap[labelmap == 254] = 255

        if self.configer.get('data.label_list', default=None) is not None:
            shape = labelmap.shape
            encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.uint8) * 255
            for i in range(len(self.configer.get('data', 'label_list'))):
                class_id = self.configer.get('data', 'label_list')[i]
                encoded_labelmap[labelmap == class_id] = i

            labelmap = encoded_labelmap

        return labelmap

    def evaluate(self, pred_dir, gt_dir):
        img_cnt = 0
        for filename in os.listdir(pred_dir):
            pred_path = os.path.join(pred_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            predmap = ImageHelper.to_np(ImageHelper.read_image(pred_path, tool='pil', mode='P'))
            gtmap = ImageHelper.to_np(ImageHelper.read_image(gt_path, tool='pil', mode='P'))
            predmap = self.relabel(np.copy(predmap))
            gtmap = self.relabel(np.copy(gtmap))

            self.seg_running_score.update(predmap[np.newaxis, :, :], gtmap[np.newaxis, :, :])
            img_cnt += 1

        Log.info('Evaluate {} images'.format(img_cnt))
        Log.info('Class mIOU: {}'.format(self.seg_running_score.get_cls_iou()))
        Log.info('mIOU: {}'.format(self.seg_running_score.get_mean_iou()))
        Log.info('Pixel ACC: {}'.format(self.seg_running_score.get_pixel_acc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None, type=str,
                        dest='config_file', help='The configs file of pose.')
    parser.add_argument('--gt_dir', default=None, type=str,
                        dest='gt_dir', help='The groundtruth annotations.')
    parser.add_argument('--pred_dir', default=None, type=str,
                        dest='pred_dir', help='The label dir of predict annotations.')
    args = parser.parse_args()

    seg_evaluator = SegEvaluator(Configer(config_file=args.config_file))
    seg_evaluator.evaluate(args.pred_dir, args.gt_dir)
