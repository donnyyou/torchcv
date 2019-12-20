#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualize the tensor of the detection.


import os
import numpy as np
import cv2
import torch
import time
from PIL import Image

from lib.data.transforms import DeNormalize
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as log


DET_DIR = 'vis/results/det'


class DetVisualizer(object):

    def __init__(self, configer):
        self.configer = configer

    def vis_bboxes(self, image_in, bboxes_list,
                   name='default', sub_dir='bbox'):
        """
          Show the diff bbox of individuals.
        """
        base_dir = os.path.join(self.configer.get('project_dir'), DET_DIR, sub_dir)

        if isinstance(image_in, Image.Image):
            image = ImageHelper.rgb2bgr(ImageHelper.to_np(image_in))

        else:
            image = image_in.copy()

        if not os.path.exists(base_dir):
            log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        img_path = os.path.join(base_dir, name if ImageHelper.is_img(name) else '{}.jpg'.format(name))

        for bbox in bboxes_list:
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.imwrite(img_path, image)

    def vis_rois(self, inputs, indices_and_rois, rois_labels=None, name='default', sub_dir='rois'):
        base_dir = os.path.join(self.configer.get('project_dir'), DET_DIR, sub_dir)

        if not os.path.exists(base_dir):
            log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        for i in range(inputs.size(0)):
            rois = indices_and_rois[indices_and_rois[:, 0] == i][:, 1:]
            ori_img = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                  mean=self.configer.get('normalize', 'mean'),
                                  std=self.configer.get('normalize', 'std'))(inputs[i])
            ori_img = ori_img.data.cpu().squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            color_num = len(self.configer.get('details', 'color_list'))

            for j in range(len(rois)):
                label = 1 if rois_labels is None else rois_labels[j]
                if label == 0:
                    continue

                class_name = self.configer.get('details', 'name_seq')[label - 1]
                cv2.rectangle(ori_img,
                                (int(rois[j][0]), int(rois[j][1])),
                                (int(rois[j][2]), int(rois[j][3])),
                                color=self.configer.get('details', 'color_list')[(label - 1) % color_num], thickness=3)
                cv2.putText(ori_img, class_name,
                            (int(rois[j][0]) + 5, int(rois[j][3]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=self.configer.get('details', 'color_list')[(label - 1) % color_num], thickness=2)

            img_path = os.path.join(base_dir, '{}_{}_{}.jpg'.format(name, i, time.time()))

            cv2.imwrite(img_path, ori_img)

    def vis_default_bboxes(self, ori_img_in, default_bboxes, labels, name='default', sub_dir='encode'):
        base_dir = os.path.join(self.configer.get('project_dir'), DET_DIR, sub_dir)

        if not os.path.exists(base_dir):
            log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        if not isinstance(ori_img_in, np.ndarray):
            ori_img = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                  mean=self.configer.get('normalize', 'mean'),
                                  std=self.configer.get('normalize', 'std'))(ori_img_in.clone())
            ori_img = ori_img.data.cpu().squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        else:
            ori_img = ori_img_in.copy()

        assert labels.size(0) == default_bboxes.size(0)

        bboxes = torch.cat([default_bboxes[:, :2] - default_bboxes[:, 2:] / 2,
                            default_bboxes[:, :2] + default_bboxes[:, 2:] / 2], 1)
        height, width, _ = ori_img.shape
        for i in range(labels.size(0)):
            if labels[i] == 0:
                continue

            class_name = self.configer.get('details', 'name_seq')[labels[i] - 1]
            color_num = len(self.configer.get('details', 'color_list'))

            cv2.rectangle(ori_img,
                          (int(bboxes[i][0] * width), int(bboxes[i][1] * height)),
                          (int(bboxes[i][2] * width), int(bboxes[i][3] * height)),
                          color=self.configer.get('details', 'color_list')[(labels[i] - 1) % color_num], thickness=3)

            cv2.putText(ori_img, class_name,
                        (int(bboxes[i][0] * width) + 5, int(bboxes[i][3] * height) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=self.configer.get('details', 'color_list')[(labels[i] - 1) % color_num], thickness=2)

        img_path = os.path.join(base_dir, '{}.jpg'.format(name))

        cv2.imwrite(img_path, ori_img)
