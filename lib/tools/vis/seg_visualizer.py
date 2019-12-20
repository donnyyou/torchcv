#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualizer for segmentation.


import os
import cv2
import numpy as np

from lib.data.transforms import DeNormalize
from lib.tools.util.logger import Logger as Log


SEG_DIR = 'vis/results/seg'


class SegVisualizer(object):

    def __init__(self, configer=None):
        self.configer = configer

    def vis_fn(self, preds, targets, ori_img_in=None, name='default', sub_dir='fn'):
        base_dir = os.path.join(self.configer.get('project_dir'), SEG_DIR, sub_dir)
        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        if not isinstance(preds, np.ndarray):
            if len(preds.size()) > 3:
                Log.error('Preds size is not valid.')
                exit(1)

            if len(preds.size()) == 3:
                preds = preds.clone().data.cpu().numpy()

            if len(preds.size()) == 2:
                preds = preds.unsqueeze(0).data.cpu().numpy()

        else:
            if len(preds.shape) > 3:
                Log.error('Preds size is not valid.')
                exit(1)

            if len(preds.shape) == 2:
                preds = preds.unsqueeze(0)

        if not isinstance(targets, np.ndarray):

            if len(targets.size()) == 3:
                targets = targets.clone().data.cpu().numpy()

            if len(targets.size()) == 2:
                targets = targets.unsqueeze(0).data.cpu().numpy()

        else:
            if len(targets.shape) == 2:
                targets = targets.unsqueeze(0)

        if ori_img_in is not None:
            if not isinstance(ori_img_in, np.ndarray):
                if len(ori_img_in.size()) < 3:
                    Log.error('Image size is not valid.')
                    exit(1)

                if len(ori_img_in.size()) == 4:
                    ori_img_in = ori_img_in.data.cpu()

                if len(ori_img_in.size()) == 3:
                    ori_img_in = ori_img_in.unsqueeze(0).data.cpu()

                ori_img = ori_img_in.clone()
                for i in range(ori_img_in.size(0)):
                    ori_img[i] = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                             mean=self.configer.get('normalize', 'mean'),
                                             std=self.configer.get('normalize', 'std'))(ori_img_in.clone())

                ori_img = ori_img.numpy().transpose(2, 3, 1).astype(np.uint8)

            else:
                if len(ori_img_in.shape) == 3:
                    ori_img_in = ori_img_in.unsqueeze(0)

                ori_img = ori_img_in.copy()

        for img_id in range(preds.shape[0]):
            label = targets[img_id]
            pred = preds[img_id]
            result = np.zeros(shape=(pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

            for i in range(self.configer.get('data', 'num_classes')):
                mask0 = np.zeros_like(label, dtype=np.uint8)
                mask1 = np.zeros_like(label, dtype=np.uint8)
                mask0[label[:] == i] += 1
                mask0[pred[:] == i] += 1
                mask1[pred[:] == i] += 1
                result[mask0[:] == 1] = self.configer.get('details', 'color_list')[i]
                result[mask1[:] == 1] = (0, 0, 0)

            image_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            if ori_img_in is not None:
                image_result  = cv2.addWeighted(ori_img[i], 0.6, image_result, 0.4, 0)

            cv2.imwrite(os.path.join(base_dir, '{}_{}.jpg'.format(name, img_id)), image_result)

    def vis_fp(self, preds, targets, ori_img_in=None, name='default', sub_dir='fp'):
        base_dir = os.path.join(self.configer.get('project_dir'), SEG_DIR, sub_dir)
        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        if not isinstance(preds, np.ndarray):
            if len(preds.size()) > 3:
                Log.error('Preds size is not valid.')
                exit(1)

            if len(preds.size()) == 3:
                preds = preds.clone().data.cpu().numpy()

            if len(preds.size()) == 2:
                preds = preds.unsqueeze(0).data.cpu().numpy()

        else:
            if len(preds.shape) > 3:
                Log.error('Preds size is not valid.')
                exit(1)

            if len(preds.shape) == 2:
                preds = preds.unsqueeze(0)

        if not isinstance(targets, np.ndarray):

            if len(targets.size()) == 3:
                targets = targets.clone().data.cpu().numpy()

            if len(targets.size()) == 2:
                targets = targets.unsqueeze(0).data.cpu().numpy()

        else:
            if len(targets.shape) == 2:
                targets = targets.unsqueeze(0)

        if ori_img_in is not None:
            if not isinstance(ori_img_in, np.ndarray):
                if len(ori_img_in.size()) < 3:
                    Log.error('Image size is not valid.')
                    exit(1)

                if len(ori_img_in.size()) == 4:
                    ori_img_in = ori_img_in.data.cpu()

                if len(ori_img_in.size()) == 3:
                    ori_img_in = ori_img_in.unsqueeze(0).data.cpu()

                ori_img = ori_img_in.clone()
                for i in range(ori_img_in.size(0)):
                    ori_img[i] = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                             mean=self.configer.get('normalize', 'mean'),
                                             std=self.configer.get('normalize', 'std'))(ori_img_in.clone())

                ori_img = ori_img.numpy().transpose(2, 3, 1).astype(np.uint8)

            else:
                if len(ori_img_in.shape) == 3:
                    ori_img_in = ori_img_in.unsqueeze(0)

                ori_img = ori_img_in.copy()

        for img_id in range(preds.shape[0]):
            label = targets[img_id]
            pred = preds[img_id]
            result = np.zeros(shape=(pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

            for i in range(self.configer.get('data', 'num_classes')):
                mask0 = np.zeros_like(label, dtype=np.uint8)
                mask1 = np.zeros_like(label, dtype=np.uint8)
                mask0[label[:] == i] += 1
                mask0[pred[:] == i] += 1
                mask1[label[:] == i] += 1
                result[mask0[:] == 1] = self.configer.get('details', 'color_list')[i]
                result[mask1[:] == 1] = (0, 0, 0)

            image_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            if ori_img_in is not None:
                image_result = cv2.addWeighted(ori_img[i], 0.6, image_result, 0.4, 0)

            cv2.imwrite(os.path.join(base_dir, '{}_{}.jpg'.format(name, img_id)), image_result)

    def error_map(self, im, pred, gt):
        canvas = im.copy()
        canvas[np.where((gt - pred != [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred[np.where((gt - pred == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        canvas = cv2.addWeighted(canvas, 1.0, pred, 1.0, 0)
        # canvas = cv2.addWeighted(im, 0.3, canvas, 0.7, 0)
        canvas[np.where((gt == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        return canvas
