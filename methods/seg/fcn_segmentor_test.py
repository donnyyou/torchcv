#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


import os
import cv2
import numpy as np
import torch
from PIL import Image

from datasets.seg.data_loader import DataLoader
from datasets.test.test_data_loader import TestDataLoader
from methods.tools.blob_helper import BlobHelper
from methods.tools.runner_helper import RunnerHelper
from models.seg.model_manager import ModelManager
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log
from utils.parser.seg_parser import SegParser
from utils.visualizer.seg_visualizer import SegVisualizer
from utils.helpers.file_helper import FileHelper
from utils.helpers.dc_helper import DCHelper
from utils.helpers.tensor_helper import TensorHelper


class FCNSegmentorTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.seg_visualizer = SegVisualizer(configer)
        self.seg_parser = SegParser(configer)
        self.seg_model_manager = ModelManager(configer)
        self.seg_data_loader = DataLoader(configer)
        self.test_loader = TestDataLoader(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.seg_net = None

        self._init_model()

    def _init_model(self):
        self.seg_net = self.seg_model_manager.semantic_segmentor()
        self.seg_net = RunnerHelper.load_net(self, self.seg_net)
        self.seg_net.eval()

    def test(self, test_dir, out_dir):
        for i, data_dict in enumerate(self.test_loader.get_testloader(test_dir=test_dir)):
            total_logits = None
            if self.configer.get('test', 'mode') == 'ss_test':
                total_logits = self.ss_test(data_dict)

            elif self.configer.get('test', 'mode') == 'sscrop_test':
                total_logits = self.sscrop_test(data_dict)

            elif self.configer.get('test', 'mode') == 'ms_test':
                total_logits = self.ms_test(data_dict)

            elif self.configer.get('test', 'mode') == 'mscrop_test':
                total_logits = self.mscrop_test(data_dict)

            else:
                Log.error('Invalid test mode:{}'.format(self.configer.get('test', 'mode')))
                exit(1)

            meta_list = DCHelper.tolist(data_dict['meta'])
            img_list = DCHelper.tolist(data_dict['img'])
            for i in range(len(meta_list)):
                filename = meta_list[i]['image_path'].split('/')[-1].split('.')[0]
                label_map = np.argmax(total_logits[i], axis=-1)
                label_img = np.array(label_map, dtype=np.uint8)
                ori_img_bgr = ImageHelper.get_cv2_bgr(img_list[i][0], mode=self.configer.get('data', 'input_mode'))
                ori_img_bgr = ImageHelper.resize(ori_img_bgr, target_size=meta_list[i]['ori_img_size'])
                image_canvas = self.seg_parser.colorize(label_img, image_canvas=ori_img_bgr)
                ImageHelper.save(image_canvas, save_path=os.path.join(out_dir, 'vis/{}.png'.format(filename)))

                if self.configer.exists('data', 'label_list'):
                    label_img = self.__relabel(label_img)

                if self.configer.exists('data', 'reduce_zero_label') and self.configer.get('data', 'reduce_zero_label'):
                    label_img = label_img + 1
                    label_img = label_img.astype(np.uint8)

                label_img = Image.fromarray(label_img, 'P')
                label_path = os.path.join(out_dir, 'label/{}.png'.format(filename))
                Log.info('Label Path: {}'.format(label_path))
                ImageHelper.save(label_img, label_path)

    def ss_test(self, in_data_dict):
        data_dict = self.blob_helper.get_blob(in_data_dict, scale=1.0)
        meta_list = DCHelper.tolist(data_dict['meta'])
        results = self._predict(data_dict)
        for i, meta in enumerate(meta_list):
            results[i] = cv2.resize(results[i][:meta['border_hw'][0], :meta['border_hw'][1]],
                                    (meta['ori_img_size'][0], meta['ori_img_size'][1]), interpolation=cv2.INTER_CUBIC)

        return results

    def sscrop_test(self, in_data_dict):
        data_dict = self.blob_helper.get_blob(in_data_dict, scale=1.0)
        crop_size = self.configer.get('test', 'crop_size')
        if any(image.size()[3] > crop_size[0] and image.size()[2] > crop_size[1]
               for image in DCHelper.tolist(data_dict['img'])):
            results = self._crop_predict(data_dict, crop_size)
        else:
            results = self._predict(data_dict)

        for i, meta in enumerate(DCHelper.tolist(data_dict['meta'])):
            results[i] = cv2.resize(results[i][:meta['border_hw'][0], :meta['border_hw'][1]],
                                    (meta['ori_img_size'][0], meta['ori_img_size'][1]), interpolation=cv2.INTER_CUBIC)
        return results

    def mscrop_test(self, in_data_dict):
        total_logits = [np.zeros((meta['ori_img_size'][1], meta['ori_img_size'][0],
                                  self.configer.get('data', 'num_classes')), np.float32)
                        for meta in DCHelper.tolist(in_data_dict['meta'])]
        for scale in self.configer.get('test', 'scale_search'):
            data_dict = self.blob_helper.get_blob(in_data_dict, scale=scale)
            crop_size = self.configer.get('test', 'crop_size')
            if any(image.size()[3] > crop_size[0] and image.size()[2] > crop_size[1]
                   for image in DCHelper.tolist(data_dict['img'])):
                results = self._crop_predict(data_dict, crop_size)
            else:
                results = self._predict(data_dict)

            for i, meta in enumerate(DCHelper.tolist(data_dict['meta'])):
                total_logits[i] += cv2.resize(results[i][:meta['border_hw'][0], :meta['border_hw'][1]],
                                              (meta['ori_img_size'][0], meta['ori_img_size'][1]),
                                              interpolation=cv2.INTER_CUBIC)
        return total_logits

    def ms_test(self, in_data_dict):
        total_logits = [np.zeros((meta['ori_img_size'][1], meta['ori_img_size'][0],
                                  self.configer.get('data', 'num_classes')), np.float32)
                        for meta in DCHelper.tolist(in_data_dict['meta'])]
        for scale in self.configer.get('test', 'scale_search'):
            data_dict = self.blob_helper.get_blob(in_data_dict, scale=scale)
            results = self._predict(data_dict)
            for i, meta in enumerate(DCHelper.tolist(data_dict['meta'])):
                total_logits[i] += cv2.resize(results[i][:meta['border_hw'][0], :meta['border_hw'][1]],
                                              (meta['ori_img_size'][0], meta['ori_img_size'][1]),
                                              interpolation=cv2.INTER_CUBIC)

        for scale in self.configer.get('test', 'scale_search'):
            data_dict = self.blob_helper.get_blob(in_data_dict, scale=scale, flip=True)
            results = self._predict(data_dict)
            for i, meta in enumerate(DCHelper.tolist(data_dict['meta'])):
                result = results[i][:meta['border_hw'][0], :meta['border_hw'][1]]
                total_logits[i] += cv2.resize(result[:, ::-1],
                                              (meta['ori_img_size'][0], meta['ori_img_size'][1]),
                                              interpolation=cv2.INTER_CUBIC)
        return total_logits

    def _crop_predict(self, data_dict, crop_size):
        split_batch = list()
        height_starts_list = list()
        width_starts_list = list()
        for image in DCHelper.tolist(data_dict['img']):
            height, width = image.size()[2:]
            np_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            height_starts = self._decide_intersection(height, crop_size[1])
            width_starts = self._decide_intersection(width, crop_size[0])
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

        out_list = list()
        with torch.no_grad():
            results = self.seg_net.forward(DCHelper.todc(split_batch, samples_per_gpu=1))
            for res in results:
                out_list.append(res[-1].permute(0, 2, 3, 1).cpu().numpy())

        total_logits = [np.zeros((meta['ori_img_size'][1], meta['ori_img_size'][0],
                                  self.configer.get('data', 'num_classes')), np.float32)
                        for meta in DCHelper.tolist(data_dict['meta'])]
        for i in range(len(height_starts_list)):
            index = 0
            for height in height_starts_list[i]:
                for width in width_starts_list[i]:
                    total_logits[i][height:height+crop_size[1], width:width+crop_size[0]] += out_list[i][index]
                    index += 1

        return total_logits

    def _decide_intersection(self, total_length, crop_length):
        stride = int(crop_length * self.configer.get('test', 'crop_stride_ratio'))            # set the stride as the paper do
        times = (total_length - crop_length) // stride + 1
        cropped_starting = []
        for i in range(times):
            cropped_starting.append(stride*i)

        if total_length - cropped_starting[-1] > crop_length:
            cropped_starting.append(total_length - crop_length)  # must cover the total image

        return cropped_starting

    def _predict(self, data_dict):
        with torch.no_grad():
            out_list = list()
            results = self.seg_net.forward(data_dict['img'])
            for res in results:
                out_list.append(res[-1].squeeze(0).permute(1, 2, 0).cpu().numpy())

        return out_list

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

