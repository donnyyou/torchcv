#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


import os
import cv2
import numpy as np
import torch
from PIL import Image

from data.test.test_data_loader import TestDataLoader
from lib.runner.blob_helper import BlobHelper
from lib.runner.runner_helper import RunnerHelper
from model.seg.model_manager import ModelManager
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as Log
from lib.tools.parser.seg_parser import SegParser
from lib.tools.vis.seg_visualizer import SegVisualizer
from lib.tools.helper.dc_helper import DCHelper


class FCNSegmentorTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.seg_visualizer = SegVisualizer(configer)
        self.seg_parser = SegParser(configer)
        self.seg_model_manager = ModelManager(configer)
        self.test_loader = TestDataLoader(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.seg_net = None

        self._init_model()

    def _init_model(self):
        self.seg_net = self.seg_model_manager.get_seg_model()
        self.seg_net = RunnerHelper.load_net(self, self.seg_net)
        self.seg_net.eval()

    def test(self, test_dir, out_dir):
        for _, data_dict in enumerate(self.test_loader.get_testloader(test_dir=test_dir)):
            total_logits = None
            if self.configer.get('test', 'mode') == 'ss_test':
                total_logits = self.ss_test(data_dict)

            elif self.configer.get('test', 'mode') == 'sscrop_test':
                total_logits = self.sscrop_test(data_dict, params_dict=self.configer.get('test', 'sscrop_test'))

            elif self.configer.get('test', 'mode') == 'ms_test':
                total_logits = self.ms_test(data_dict, params_dict=self.configer.get('test', 'ms_test'))

            elif self.configer.get('test', 'mode') == 'mscrop_test':
                total_logits = self.mscrop_test(data_dict, params_dict=self.configer.get('test', 'mscrop_test'))

            else:
                Log.error('Invalid test mode:{}'.format(self.configer.get('test', 'mode')))
                exit(1)

            meta_list = DCHelper.tolist(data_dict['meta'])
            for i in range(len(meta_list)):
                label_map = np.argmax(total_logits[i], axis=-1)
                label_img = np.array(label_map, dtype=np.uint8)
                ori_img_bgr = ImageHelper.read_image(meta_list[i]['img_path'], tool='cv2', mode='BGR')
                image_canvas = self.seg_parser.colorize(label_img, image_canvas=ori_img_bgr)
                ImageHelper.save(image_canvas,
                                 save_path=os.path.join(out_dir, 'vis/{}.png'.format(meta_list[i]['filename'])))

                if self.configer.get('data.label_list', default=None) is not None:
                    label_img = self.__relabel(label_img)

                if self.configer.get('data.reduce_zero_label', default=False):
                    label_img = label_img + 1
                    label_img = label_img.astype(np.uint8)

                label_img = Image.fromarray(label_img, 'P')
                label_path = os.path.join(out_dir, 'label/{}.png'.format(meta_list[i]['filename']))
                Log.info('Label Path: {}'.format(label_path))
                ImageHelper.save(label_img, label_path)

    def ss_test(self, in_data_dict):
        data_dict = self.blob_helper.get_blob(in_data_dict, scale=1.0)
        results = self._predict(data_dict)
        return results

    def ms_test(self, in_data_dict, params_dict):
        total_logits = [np.zeros((meta['ori_img_size'][1], meta['ori_img_size'][0],
                                  self.configer.get('data', 'num_classes')), np.float32)
                        for meta in DCHelper.tolist(in_data_dict['meta'])]
        for scale in params_dict['scale_search']:
            data_dict = self.blob_helper.get_blob(in_data_dict, scale=scale)
            results = self._predict(data_dict)
            for i in range(len(total_logits)):
                total_logits[i] += results[i]

        for scale in params_dict['scale_search']:
            data_dict = self.blob_helper.get_blob(in_data_dict, scale=scale, flip=True)
            results = self._predict(data_dict)
            for i in range(len(total_logits)):
                total_logits[i] += results[i][:, ::-1]

        return total_logits

    def sscrop_test(self, in_data_dict, params_dict):
        data_dict = self.blob_helper.get_blob(in_data_dict, scale=1.0)
        if any(image.size()[2] < params_dict['crop_size'][0] or image.size()[1] < params_dict['crop_size'][1]
                   for image in DCHelper.tolist(data_dict['img'])):
            results = self._predict(data_dict)
        else:
            results = self._crop_predict(data_dict, params_dict['crop_size'], params_dict['crop_stride_ratio'])

        return results

    def mscrop_test(self, in_data_dict, params_dict):
        total_logits = [np.zeros((meta['ori_img_size'][1], meta['ori_img_size'][0],
                                  self.configer.get('data', 'num_classes')), np.float32)
                        for meta in DCHelper.tolist(in_data_dict['meta'])]
        for scale in params_dict['scale_search']:
            data_dict = self.blob_helper.get_blob(in_data_dict, scale=scale)
            if any(image.size()[2] < params_dict['crop_size'][0] or image.size()[1] < params_dict['crop_size'][1]
                   for image in DCHelper.tolist(data_dict['img'])):
                results = self._predict(data_dict)
            else:
                results = self._crop_predict(data_dict, params_dict['crop_size'], params_dict['crop_stride_ratio'])

            for i in range(len(total_logits)):
                total_logits[i] += results[i]

        for scale in params_dict['scale_search']:
            data_dict = self.blob_helper.get_blob(in_data_dict, scale=scale, flip=True)
            if any(image.size()[2] < params_dict['crop_size'][0] or image.size()[1] < params_dict['crop_size'][1]
                   for image in DCHelper.tolist(data_dict['img'])):
                results = self._predict(data_dict)
            else:
                results = self._crop_predict(data_dict, params_dict['crop_size'], params_dict['crop_stride_ratio'])

            for i in range(len(total_logits)):
                total_logits[i] += results[i][:, ::-1]

        return total_logits

    def _crop_predict(self, data_dict, crop_size, crop_stride_ratio):
        split_batch = list()
        height_starts_list = list()
        width_starts_list = list()
        hw_list = list()
        for image in DCHelper.tolist(data_dict['img']):
            height, width = image.size()[1:]
            hw_list.append([height, width])
            np_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            height_starts = self._decide_intersection(height, crop_size[1], crop_stride_ratio)
            width_starts = self._decide_intersection(width, crop_size[0], crop_stride_ratio)
            split_crops = []
            for height in height_starts:
                for width in width_starts:
                    image_crop = np_image[height:height + crop_size[1], width:width + crop_size[0]]
                    split_crops.append(image_crop[np.newaxis, :])

            height_starts_list.append(height_starts)
            width_starts_list.append(width_starts)
            split_crops = np.concatenate(split_crops, axis=0)  # (n, crop_image_size, crop_image_size, 3)
            inputs = torch.from_numpy(split_crops).permute(0, 3, 1, 2).to(self.device)
            split_batch.append(inputs)

        assert len(split_batch) == torch.cuda.device_count(), 'Only support one image per gpu.'
        out_list = list()
        with torch.no_grad():
            results = self.seg_net(dict(img=DCHelper.todc(split_batch, stack=False, samples_per_gpu=True, concat=True)))
            results = results if isinstance(results, (list, tuple)) else [results]
            for res in results:
                out_list.append(res['out'].permute(0, 2, 3, 1).cpu().numpy())

        total_logits = [np.zeros((hw[0], hw[1],
                                  self.configer.get('data', 'num_classes')), np.float32) for hw in hw_list]
        count_predictions = [np.zeros((hw[0], hw[1],
                                       self.configer.get('data', 'num_classes')), np.float32) for hw in hw_list]
        for i in range(len(height_starts_list)):
            index = 0
            for height in height_starts_list[i]:
                for width in width_starts_list[i]:
                    total_logits[i][height:height+crop_size[1], width:width+crop_size[0]] += out_list[i][index]
                    count_predictions[i][height:height+crop_size[1], width:width+crop_size[0]] += 1
                    index += 1

        for i in range(len(total_logits)):
            total_logits[i] /= count_predictions[i]

        for i, meta in enumerate(DCHelper.tolist(data_dict['meta'])):
            total_logits[i] = cv2.resize(total_logits[i][:meta['border_wh'][1], :meta['border_wh'][0]],
                                         tuple(meta['ori_img_size']), interpolation=cv2.INTER_CUBIC)

        return total_logits

    def _decide_intersection(self, total_length, crop_length, crop_stride_ratio):
        stride = int(crop_length * crop_stride_ratio)            # set the stride as the paper do
        times = (total_length - crop_length) // stride + 1
        cropped_starting = []
        for i in range(times):
            cropped_starting.append(stride*i)

        if total_length - cropped_starting[-1] > crop_length:
            cropped_starting.append(total_length - crop_length)  # must cover the total image

        return cropped_starting

    def _predict(self, data_dict):
        with torch.no_grad():
            total_logits = list()
            results = self.seg_net(data_dict)
            results = results if isinstance(results, (list, tuple)) else [results]
            for res in results:
                assert res['out'].size(0) == 1, 'Only support one image per gpu.'
                total_logits.append(res['out'].squeeze(0).permute(1, 2, 0).cpu().numpy())

            for i, meta in enumerate(DCHelper.tolist(data_dict['meta'])):
                total_logits[i] = cv2.resize(total_logits[i][:meta['border_wh'][1], :meta['border_wh'][0]],
                                             tuple(meta['ori_img_size']), interpolation=cv2.INTER_CUBIC)

        return total_logits

    def __relabel(self, label_map):
        height, width = label_map.shape
        label_dst = np.zeros((height, width), dtype=np.uint8)
        for i in range(self.configer.get('data', 'num_classes')):
            label_dst[label_map == i] = self.configer.get('data', 'label_list')[i]

        label_dst = np.array(label_dst, dtype=np.uint8)

        return label_dst


