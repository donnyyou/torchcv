#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (donnyyou@163.com)
# Class Definition for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
from PIL import Image

from datasets.seg_data_loader import SegDataLoader
from datasets.tools.data_transformer import DataTransformer
from methods.tools.blob_helper import BlobHelper
from methods.tools.module_utilizer import ModuleUtilizer
from models.seg_model_manager import SegModelManager
from utils.helpers.file_helper import FileHelper
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
        self.seg_model_manager = SegModelManager(configer)
        self.seg_data_loader = SegDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.data_transformer = DataTransformer(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.seg_net = None

        self._init_model()

    def _init_model(self):
        self.seg_net = self.seg_model_manager.semantic_segmentor()
        self.seg_net = self.module_utilizer.load_net(self.seg_net)
        self.seg_net.eval()

    def __test_img(self, image_path, label_path, vis_path, raw_path):
        Log.info('Image Path: {}'.format(image_path))
        ori_image = ImageHelper.read_image(image_path,
                                           tool=self.configer.get('data', 'image_tool'),
                                           mode=self.configer.get('data', 'input_mode'))

        ori_width, ori_height = ImageHelper.get_size(ori_image)

        total_logits = np.zeros((ori_height, ori_width, self.configer.get('data', 'num_classes')), np.float32)
        for scale in self.configer.get('test', 'scale_search'):
            image = self.blob_helper.make_input(image=ori_image,
                                                input_size=self.configer.get('test', 'input_size'),
                                                scale=scale)
            if self.configer.get('test', 'crop_test'):
                crop_size = self.configer.get('test', 'crop_size')
                if image.size()[3] > crop_size[0] and image.size()[2] > crop_size[1]:
                    results = self._crop_predict(image, crop_size)
                else:
                    results = self._predict(image)
            else:
                results = self._predict(image)

            results = cv2.resize(results, (ori_width, ori_height), interpolation=cv2.INTER_LINEAR)
            total_logits += results

        if self.configer.get('test', 'mirror'):
            if self.configer.get('data', 'image_tool') == 'cv2':
                image = cv2.flip(ori_image, 1)
            else:
                image = ori_image.transpose(Image.FLIP_LEFT_RIGHT)

            image = self.blob_helper.make_input(image, input_size=self.configer.get('test', 'input_size'), scale=1.0)
            if self.configer.get('test', 'crop_test'):
                crop_size = self.configer.get('test', 'crop_size')
                if image.size()[3] > crop_size[0] and image.size()[2] > crop_size[1]:
                    results = self._crop_predict(image, crop_size)
                else:
                    results = self._predict(image)
            else:
                results = self._predict(image)

            results = cv2.resize(results[:, ::-1], (ori_width, ori_height), interpolation=cv2.INTER_LINEAR)
            total_logits += results

        label_map = np.argmax(total_logits, axis=-1)
        label_img = np.array(label_map, dtype=np.uint8)
        image_bgr = cv2.cvtColor(np.array(ori_image), cv2.COLOR_RGB2BGR)
        image_canvas = self.seg_parser.colorize(label_img, image_canvas=image_bgr)
        ImageHelper.save(image_canvas, save_path=vis_path)
        ImageHelper.save(ori_image, save_path=raw_path)

        if not self.configer.is_empty('data', 'label_list'):
            label_img = self.__relabel(label_img)

        label_img = Image.fromarray(label_img, 'P')
        Log.info('Label Path: {}'.format(label_path))
        ImageHelper.save(label_img, label_path)

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
            results = results[0].permute(0, 2, 3, 1).cpu().numpy()

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
            results = results[0].squeeze().permute(1, 2, 0).cpu().numpy()

        return results

    def __relabel(self, label_map):
        height, width = label_map.shape
        label_dst = np.zeros((height, width), dtype=np.uint8)
        for i in range(self.configer.get('data', 'num_classes')):
            label_dst[label_map == i] = self.configer.get('data', 'label_list')[i]

        label_dst = np.array(label_dst, dtype=np.uint8)

        return label_dst

    def test(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/seg', self.configer.get('dataset'))

        test_img = self.configer.get('test_img')
        test_dir = self.configer.get('test_dir')
        if test_img is None and test_dir is None:
            Log.error('test_img & test_dir not exists.')
            exit(1)

        if test_img is not None and test_dir is not None:
            Log.error('Either test_img or test_dir.')
            exit(1)

        if test_img is not None:
            base_dir = os.path.join(base_dir, 'test_img')
            filename = test_img.rstrip().split('/')[-1]
            label_path = os.path.join(base_dir, 'label', '{}.png'.format('.'.join(filename.split('.')[:-1])))
            raw_path = os.path.join(base_dir, 'raw', filename)
            vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
            FileHelper.make_dirs(label_path, is_file=True)
            FileHelper.make_dirs(raw_path, is_file=True)
            FileHelper.make_dirs(vis_path, is_file=True)

            self.__test_img(test_img, label_path, vis_path, raw_path)

        else:
            base_dir = os.path.join(base_dir, 'test_dir', test_dir.rstrip('/').split('/')[-1])
            FileHelper.make_dirs(base_dir)

            for filename in FileHelper.list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                label_path = os.path.join(base_dir, 'label', '{}.png'.format('.'.join(filename.split('.')[:-1])))
                raw_path = os.path.join(base_dir, 'raw', filename)
                vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
                FileHelper.make_dirs(label_path, is_file=True)
                FileHelper.make_dirs(raw_path, is_file=True)
                FileHelper.make_dirs(vis_path, is_file=True)

                self.__test_img(image_path, label_path, vis_path, raw_path)

    def debug(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/seg', self.configer.get('dataset'), 'debug')

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

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
                cv2.imwrite(os.path.join(base_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()

