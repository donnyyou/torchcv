#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Image Classifier.


import os
import cv2
import json
import torch
import torchvision.transforms as transforms
from PIL import Image

from lib.runner.blob_helper import BlobHelper
from lib.runner.runner_helper import RunnerHelper
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.helper.json_helper import JsonHelper
from lib.tools.util.logger import Logger as Log
from lib.tools.parser.cls_parser import ClsParser
from model.cls.model_manager import ModelManager
from data.cls.data_loader import DataLoader


class ImageClassifierTest(object):

    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.cls_model_manager = ModelManager(configer)
        self.cls_data_loader = DataLoader(configer)
        self.cls_parser = ClsParser(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.cls_net = None
        if self.configer.get('dataset') == 'imagenet':
            with open(os.path.join(self.configer.get('project_dir'),
                                   'datasets/cls/imagenet/imagenet_class_index.json')) as json_stream:
                name_dict = json.load(json_stream)
                name_seq = [name_dict[str(i)][1] for i in range(self.configer.get('data', 'num_classes'))]
                self.configer.add(['details', 'name_seq'], name_seq)

        self._init_model()

    def _init_model(self):
        self.cls_net = self.cls_model_manager.get_cls_model()
        self.cls_net = RunnerHelper.load_net(self, self.cls_net)
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

        _, pred = outputs.topk(maxk, 0, True, True)
        for k in topk:
            if k == 1:
                json_dict['label'] = pred[0]

            else:
                json_dict['label_top{}'.format(k)] = pred[:k]

        return json_dict

    def debug(self, vis_dir):
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

                json_dict = self.__get_info_tree(labels_target[j])
                image_canvas = self.cls_parser.draw_label(ori_img_bgr.copy(), json_dict['label'])

                cv2.imwrite(os.path.join(vis_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()
