#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
from PIL import Image

from datasets.seg.data_loader import DataLoader
from methods.tools.blob_helper import BlobHelper
from methods.tools.runner_helper import RunnerHelper
from models.seg.model_manager import ModelManager
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log
from vis.parser.seg_parser import SegParser
from vis.visualizer.seg_visualizer import SegVisualizer


class FCNSegmentorTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.seg_visualizer = SegVisualizer(configer)
        self.seg_parser = SegParser(configer)
        self.seg_model_manager = ModelManager(configer)
        self.seg_data_loader = DataLoader(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.seg_net = None

        self._init_model()

    def _init_model(self):
        self.seg_net = self.seg_model_manager.semantic_segmentor()
        self.seg_net = RunnerHelper.load_net(self, self.seg_net)
        self.seg_net.eval()

    def _get_blob(self, ori_image, scale=None):
        assert scale is not None
        image = None
        if self.configer.exists('test', 'input_size'):
            image = self.blob_helper.make_input(image=ori_image,
                                                input_size=self.configer.get('test', 'input_size'),
                                                scale=scale)

        elif self.configer.exists('test', 'min_side_length') and not self.configer.exists('test', 'max_side_length'):
            image = self.blob_helper.make_input(image=ori_image,
                                                min_side_length=self.configer.get('test', 'min_side_length'),
                                                scale=scale)

        elif not self.configer.exists('test', 'min_side_length') and self.configer.exists('test', 'max_side_length'):
            image = self.blob_helper.make_input(image=ori_image,
                                                max_side_length=self.configer.get('test', 'max_side_length'),
                                                scale=scale)

        elif self.configer.exists('test', 'min_side_length') and self.configer.exists('test', 'max_side_length'):
            image = self.blob_helper.make_input(image=ori_image,
                                                min_side_length=self.configer.get('test', 'min_side_length'),
                                                max_side_length=self.configer.get('test', 'max_side_length'),
                                                scale=scale)

        else:
            Log.error('Test setting error')
            exit(1)

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

    def test_img(self, image_path, label_path, vis_path, raw_path):
        Log.info('Image Path: {}'.format(image_path))
        ori_image = ImageHelper.read_image(image_path,
                                           tool=self.configer.get('data', 'image_tool'),
                                           mode=self.configer.get('data', 'input_mode'))
        total_logits = None
        if self.configer.get('test', 'mode') == 'ss_test':
            total_logits = self.ss_test(ori_image)

        elif self.configer.get('test', 'mode') == 'sscrop_test':
            total_logits = self.sscrop_test(ori_image)

        elif self.configer.get('test', 'mode') == 'ms_test':
            total_logits = self.ms_test(ori_image)

        elif self.configer.get('test', 'mode') == 'mscrop_test':
            total_logits = self.mscrop_test(ori_image)

        else:
            Log.error('Invalid test mode:{}'.format(self.configer.get('test', 'mode')))
            exit(1)

        label_map = np.argmax(total_logits, axis=-1)
        label_img = np.array(label_map, dtype=np.uint8)
        ori_img_bgr = ImageHelper.get_cv2_bgr(ori_image, mode=self.configer.get('data', 'input_mode'))
        image_canvas = self.seg_parser.colorize(label_img, image_canvas=ori_img_bgr)
        ImageHelper.save(image_canvas, save_path=vis_path)
        ImageHelper.save(ori_image, save_path=raw_path)

        if self.configer.exists('data', 'label_list'):
            label_img = self.__relabel(label_img)

        if self.configer.exists('data', 'reduce_zero_label') and self.configer.get('data', 'reduce_zero_label'):
            label_img = label_img + 1
            label_img = label_img.astype(np.uint8)

        label_img = Image.fromarray(label_img, 'P')
        Log.info('Label Path: {}'.format(label_path))
        ImageHelper.save(label_img, label_path)

    def ss_test(self, ori_image):
        ori_width, ori_height = ImageHelper.get_size(ori_image)
        total_logits = np.zeros((ori_height, ori_width, self.configer.get('data', 'num_classes')), np.float32)
        image, border_hw = self._get_blob(ori_image, scale=1.0)
        results = self._predict(image)
        results = cv2.resize(results[:border_hw[0], :border_hw[1]],
                             (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)
        total_logits += results
        return total_logits

    def sscrop_test(self, ori_image):
        ori_width, ori_height = ImageHelper.get_size(ori_image)
        total_logits = np.zeros((ori_height, ori_width, self.configer.get('data', 'num_classes')), np.float32)
        image, _ = self._get_blob(ori_image, scale=1.0)
        crop_size = self.configer.get('test', 'crop_size')
        if image.size()[3] > crop_size[0] and image.size()[2] > crop_size[1]:
            results = self._crop_predict(image, crop_size)
        else:
            results = self._predict(image)

        results = cv2.resize(results, (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)
        total_logits += results
        return total_logits

    def mscrop_test(self, ori_image):
        ori_width, ori_height = ImageHelper.get_size(ori_image)
        total_logits = np.zeros((ori_height, ori_width, self.configer.get('data', 'num_classes')), np.float32)
        for scale in self.configer.get('test', 'scale_search'):
            image, _ = self._get_blob(ori_image, scale=scale)
            crop_size = self.configer.get('test', 'crop_size')
            if image.size()[3] > crop_size[0] and image.size()[2] > crop_size[1]:
                results = self._crop_predict(image, crop_size)
            else:
                results = self._predict(image)

            results = cv2.resize(results, (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)
            total_logits += results

        return total_logits

    def ms_test(self, ori_image):
        ori_width, ori_height = ImageHelper.get_size(ori_image)
        total_logits = np.zeros((ori_height, ori_width, self.configer.get('data', 'num_classes')), np.float32)
        for scale in self.configer.get('test', 'scale_search'):
            image, border_hw = self._get_blob(ori_image, scale=scale)
            results = self._predict(image)
            results = cv2.resize(results[:border_hw[0], :border_hw[1]],
                                 (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)
            total_logits += results

        if self.configer.get('data', 'image_tool') == 'cv2':
            mirror_image = cv2.flip(ori_image, 1)
        else:
            mirror_image = ori_image.transpose(Image.FLIP_LEFT_RIGHT)

        image, border_hw = self._get_blob(mirror_image, scale=1.0)
        results = self._predict(image)
        results = results[:border_hw[0], :border_hw[1]]
        results = cv2.resize(results[:, ::-1], (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)
        total_logits += results
        return total_logits

    def _crop_predict(self, image, crop_size):
        height, width = image.size()[2:]
        np_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        height_starts = self._decide_intersection(height, crop_size[1])
        width_starts = self._decide_intersection(width, crop_size[0])
        split_crops = []
        for height in height_starts:
            for width in width_starts:
                image_crop = np_image[height:height + crop_size[1], width:width + crop_size[0]]
                split_crops.append(image_crop[np.newaxis, :])

        split_crops = np.concatenate(split_crops, axis=0)  # (n, crop_image_size, crop_image_size, 3)
        inputs = torch.from_numpy(split_crops).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            results = self.seg_net.forward(inputs)
            results = results[-1].permute(0, 2, 3, 1).cpu().numpy()

        reassemble = np.zeros((np_image.shape[0], np_image.shape[1], results.shape[-1]), np.float32)
        index = 0
        for height in height_starts:
            for width in width_starts:
                reassemble[height:height+crop_size[1], width:width+crop_size[0]] += results[index]
                index += 1

        return reassemble

    def _decide_intersection(self, total_length, crop_length):
        stride = int(crop_length * self.configer.get('test', 'crop_stride_ratio'))            # set the stride as the paper do
        times = (total_length - crop_length) // stride + 1
        cropped_starting = []
        for i in range(times):
            cropped_starting.append(stride*i)

        if total_length - cropped_starting[-1] > crop_length:
            cropped_starting.append(total_length - crop_length)  # must cover the total image

        return cropped_starting

    def _predict(self, inputs):
        with torch.no_grad():
            results = self.seg_net.forward(inputs)
            results = results[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()

        return results

    def __relabel(self, label_map):
        height, width = label_map.shape
        label_dst = np.zeros((height, width), dtype=np.uint8)
        for i in range(self.configer.get('data', 'num_classes')):
            label_dst[label_map == i] = self.configer.get('data', 'label_list')[i]

        label_dst = np.array(label_dst, dtype=np.uint8)

        return label_dst

    def debug(self, vis_dir):
        count = 0
        for i, data_dict in enumerate(self.seg_data_loader.get_trainloader()):
            inputs = data_dict['img']
            targets = data_dict['labelmap']
            for j in range(inputs.size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                image_bgr = self.blob_helper.tensor2bgr(inputs[j])
                label_map = targets[j].numpy()
                image_canvas = self.seg_parser.colorize(label_map, image_canvas=image_bgr)
                cv2.imwrite(os.path.join(vis_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()

