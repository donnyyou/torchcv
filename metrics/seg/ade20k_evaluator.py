#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import os
import numpy as np
import argparse

from metrics.seg.seg_running_score import SegRunningScore
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log
from utils.tools.configer import Configer


class ADE20KEvaluator(object):
    def __init__(self, configer):
        self.configer = configer
        self.seg_running_score = SegRunningScore(configer)

    def relabel(self, labelmap):
        if self.configer.get('data', 'reduce_zero_label'):
            labelmap[labelmap == 0] = 255
            labelmap = (labelmap - 1).astype(np.uint8)
            labelmap[labelmap == 254] = 255

        return labelmap

    def evaluate(self, pred_dir, gt_dir):
        img_cnt = 0
        for filename in os.listdir(pred_dir):
            pred_path = os.path.join(pred_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            predmap = ImageHelper.img2np(ImageHelper.read_image(pred_path, tool='pil', mode='P'))
            gtmap = ImageHelper.img2np(ImageHelper.read_image(gt_path, tool='pil', mode='P'))
            predmap = self.relabel(predmap)
            gtmap = self.relabel(gtmap)

            self.seg_running_score.update(predmap[np.newaxis, :, :], gtmap[np.newaxis, :, :])
            img_cnt += 1

        Log.info('Evaluate {} images'.format(img_cnt))
        Log.info('mIOU: {}'.format(self.seg_running_score.get_mean_iou()))
        Log.info('Pixel ACC: {}'.format(self.seg_running_score.get_pixel_acc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes_file', default=None, type=str,
                        dest='hypes_file', help='The hypes file of pose.')
    parser.add_argument('--gt_dir', default=None, type=str,
                        dest='gt_dir', help='The groundtruth annotations.')
    parser.add_argument('--pred_dir', default=None, type=str,
                        dest='pred_dir', help='The label dir of predict annotations.')
    args = parser.parse_args()

    ade20k_evaluator = ADE20KEvaluator(Configer(hypes_file=args.hypes_file))
    ade20k_evaluator.evaluate(args.pred_dir, args.gt_dir)
