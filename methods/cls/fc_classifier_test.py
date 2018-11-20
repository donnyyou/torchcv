#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (donnyyou@163.com)
# Class Definition for Image Classifier.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import json
import torch
import torchvision.transforms as transforms
from PIL import Image

from datasets.cls_data_loader import ClsDataLoader
from methods.tools.blob_helper import BlobHelper
from methods.tools.module_utilizer import ModuleUtilizer
from models.cls_model_manager import ClsModelManager
from utils.helpers.file_helper import FileHelper
from utils.helpers.image_helper import ImageHelper
from utils.helpers.json_helper import JsonHelper
from utils.tools.logger import Logger as Log
from vis.parser.cls_parser import ClsParser


class FCClassifierTest(object):

    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.cls_model_manager = ClsModelManager(configer)
        self.cls_data_loader = ClsDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.cls_parser = ClsParser(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.cls_net = None
        if self.configer.get('dataset') == 'imagenet':
            with open(os.path.join(self.configer.get('project_dir'),
                                   'datasets/cls/imagenet/imagenet_class_index.json')) as json_stream:
                name_dict = json.load(json_stream)
                name_seq = [name_dict[str(i)][1] for i in range(self.configer.get('data', 'num_classes'))]
                self.configer.add_key_value(['details', 'name_seq'], name_seq)

        self._init_model()

    def _init_model(self):
        self.cls_net = self.cls_model_manager.image_classifier()
        self.cls_net = self.module_utilizer.load_net(self.cls_net)
        self.cls_net.eval()

    def __test_img(self, image_path, json_path, raw_path, vis_path):
        Log.info('Image Path: {}'.format(image_path))
        img = ImageHelper.read_image(image_path,
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))

        trans = None
        if self.configer.get('dataset') == 'imagenet':
            if self.configer.get('data', 'image_tool') == 'cv2':
                img = Image.fromarray(img)

            trans = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
            ])

        assert trans is not None
        img = trans(img)

        ori_img_bgr = ImageHelper.get_cv2_bgr(img, mode=self.configer.get('data', 'input_mode'))

        inputs = self.blob_helper.make_input(img,
                                             input_size=self.configer.get('test', 'input_size'), scale=1.0)

        with torch.no_grad():
            outputs = self.cls_net(inputs)

        json_dict = self.__get_info_tree(outputs, image_path)

        image_canvas = self.cls_parser.draw_label(ori_img_bgr.copy(), json_dict['label'])
        cv2.imwrite(vis_path, image_canvas)
        cv2.imwrite(raw_path, ori_img_bgr)

        Log.info('Json Path: {}'.format(json_path))
        JsonHelper.save_file(json_dict, json_path)
        return json_dict

    def __get_info_tree(self, outputs, image_path=None):
        json_dict = dict()
        if image_path is not None:
            json_dict['image_path'] = image_path

        topk = (1, 3, 5)
        maxk = max(topk)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        for k in topk:
            if k == 1:
                json_dict['label'] = pred[0][0]

            else:
                json_dict['label_top{}'.format(k)] = pred[0][:k]

        return json_dict

    def test(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/cls', self.configer.get('dataset'))

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
            json_path = os.path.join(base_dir, 'json', '{}.json'.format('.'.join(filename.split('.')[:-1])))
            raw_path = os.path.join(base_dir, 'raw', filename)
            vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
            FileHelper.make_dirs(json_path, is_file=True)
            FileHelper.make_dirs(raw_path, is_file=True)
            FileHelper.make_dirs(vis_path, is_file=True)

            self.__test_img(test_img, json_path, raw_path, vis_path)

        else:
            base_dir = os.path.join(base_dir, 'test_dir', test_dir.rstrip('/').split('/')[-1])
            FileHelper.make_dirs(base_dir)

            for filename in FileHelper.list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                json_path = os.path.join(base_dir, 'json', '{}.json'.format('.'.join(filename.split('.')[:-1])))
                raw_path = os.path.join(base_dir, 'raw', filename)
                vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
                FileHelper.make_dirs(json_path, is_file=True)
                FileHelper.make_dirs(raw_path, is_file=True)
                FileHelper.make_dirs(vis_path, is_file=True)

                self.__test_img(image_path, json_path, raw_path, vis_path)

    def debug(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/cls', self.configer.get('dataset'), 'debug')

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        count = 0
        for i, data_dict in enumerate(self.cls_data_loader.get_trainloader()):
            inputs = data_dict['img']
            labels = data_dict['label']
            eye_matrix = torch.eye(self.configer.get('data', 'num_classes'))
            labels_target = eye_matrix[labels.view(-1)].view(inputs.size(0), self.configer.get('data', 'num_classes'))

            for j in range(inputs.size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                ori_img_bgr = self.blob_helper.tensor2bgr(inputs[j])

                json_dict = self.__get_info_tree(labels_target)
                image_canvas = self.cls_parser.draw_label(ori_img_bgr.copy(), json_dict['label'])

                cv2.imwrite(os.path.join(base_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()