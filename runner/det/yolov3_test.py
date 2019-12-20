#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


import os
import torch

from data.test.test_data_loader import TestDataLoader
from lib.runner.runner_helper import RunnerHelper
from model.det.model_manager import ModelManager
from lib.tools.helper.det_helper import DetHelper
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.helper.json_helper import JsonHelper
from lib.tools.helper.dc_helper import DCHelper
from lib.tools.util.logger import Logger as Log
from lib.tools.parser.det_parser import DetParser
from lib.tools.vis.det_visualizer import DetVisualizer


class YOLOv3Test(object):
    def __init__(self, configer):
        self.configer = configer
        self.det_visualizer = DetVisualizer(configer)
        self.det_parser = DetParser(configer)
        self.det_model_manager = ModelManager(configer)
        self.test_loader = TestDataLoader(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.det_net = None

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = RunnerHelper.load_net(self, self.det_net)
        self.det_net.eval()

    def test(self, test_dir, out_dir):
        for _, data_dict in enumerate(self.test_loader.get_testloader(test_dir=test_dir)):
            data_dict['testing'] = True
            detections = self.det_net(data_dict)
            meta_list = DCHelper.tolist(data_dict['meta'])
            batch_detections = self.decode(detections, self.configer, meta_list)
            for i in range(len(meta_list)):
                ori_img_bgr = ImageHelper.read_image(meta_list[i]['img_path'], tool='cv2', mode='BGR')
                json_dict = self.__get_info_tree(batch_detections[i])
                image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(), json_dict,
                                                           conf_threshold=self.configer.get('res', 'vis_conf_thre'))
                ImageHelper.save(image_canvas,
                                 save_path=os.path.join(out_dir, 'vis/{}.png'.format(meta_list[i]['filename'])))

                Log.info('Json Path: {}'.format(os.path.join(out_dir, 'json/{}.json'.format(meta_list[i]['filename']))))
                JsonHelper.save_file(json_dict,
                                     save_path=os.path.join(out_dir, 'json/{}.json'.format(meta_list[i]['filename'])))

    @staticmethod
    def decode(batch_detections, configer, meta):
        output = [None for _ in range(len(meta))]
        for i in range(len(meta)):
            image_pred = batch_detections[i]
            image_pred[:, 0] *= meta[i]['ori_img_size'][0]
            image_pred[:, 1] *= meta[i]['ori_img_size'][1]
            image_pred[:, 2] *= meta[i]['ori_img_size'][0]
            image_pred[:, 3] *= meta[i]['ori_img_size'][1]
            # Filter out confidence scores below threshold
            image_pred = image_pred[image_pred[:, 4] > configer.get('res', 'val_conf_thre')]
            # If none are remaining => process next image
            if image_pred.numel() == 0:
                continue

            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + configer.get('data', 'num_classes')], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            valid_ind = DetHelper.cls_nms(detections[:, :5], labels=class_pred.squeeze(1),
                                          max_threshold=configer.get('res', 'nms')['max_threshold'], return_ind=True)
            output[i] = detections[valid_ind]

        return output

    def __get_info_tree(self, detections):
        json_dict = dict()
        object_list = list()
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                object_dict = dict()
                object_dict['bbox'] = [x1.item(), y1.item(), x2.item(), y2.item()]
                object_dict['label'] = int(cls_pred.item())
                object_dict['score'] = float('%.2f' % conf.item())
                object_list.append(object_dict)

        json_dict['objects'] = object_list

        return json_dict
