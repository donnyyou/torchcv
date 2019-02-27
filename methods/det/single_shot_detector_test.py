#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


import os
import cv2
import torch
import torch.nn.functional as F

from datasets.det.data_loader import DataLoader
from methods.tools.blob_helper import BlobHelper
from methods.tools.runner_helper import RunnerHelper
from models.det.model_manager import ModelManager
from utils.helpers.det_helper import DetHelper
from utils.helpers.image_helper import ImageHelper
from utils.helpers.json_helper import JsonHelper
from models.det.layers.ssd_priorbox_layer import SSDPriorBoxLayer
from models.det.layers.ssd_target_generator import SSDTargetGenerator
from utils.tools.logger import Logger as Log
from utils.parser.det_parser import DetParser
from utils.visualizer.det_visualizer import DetVisualizer


class SingleShotDetectorTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.det_visualizer = DetVisualizer(configer)
        self.det_parser = DetParser(configer)
        self.det_model_manager = ModelManager(configer)
        self.det_data_loader = DataLoader(configer)
        self.ssd_priorbox_layer = SSDPriorBoxLayer(configer)
        self.ssd_target_generator = SSDTargetGenerator(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.det_net = None

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = RunnerHelper.load_net(self, self.det_net)
        self.det_net.eval()

    def __test_img(self, image_path, json_path, raw_path, vis_path):
        Log.info('Image Path: {}'.format(image_path))
        img = ImageHelper.read_image(image_path,
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        ori_img_bgr = ImageHelper.get_cv2_bgr(img, mode=self.configer.get('data', 'input_mode'))

        inputs = self.blob_helper.make_input(img,
                                             input_size=self.configer.get('test', 'input_size'), scale=1.0)

        with torch.no_grad():
            feat_list, bbox, cls = self.det_net(inputs)

        batch_detections = self.decode(bbox, cls,
                                       self.ssd_priorbox_layer(feat_list, self.configer.get('test', 'input_size')),
                                       self.configer, [inputs.size(3), inputs.size(2)])
        json_dict = self.__get_info_tree(batch_detections[0], ori_img_bgr, [inputs.size(3), inputs.size(2)])

        image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                   json_dict,
                                                   conf_threshold=self.configer.get('res', 'vis_conf_thre'))
        cv2.imwrite(vis_path, image_canvas)
        cv2.imwrite(raw_path, ori_img_bgr)

        Log.info('Json Path: {}'.format(json_path))
        JsonHelper.save_file(json_dict, json_path)
        return json_dict

    @staticmethod
    def decode(bbox, conf, default_boxes, configer, input_size):
        loc = bbox
        if configer.get('phase') != 'debug':
            conf = F.softmax(conf, dim=-1)

        default_boxes = default_boxes.unsqueeze(0).repeat(loc.size(0), 1, 1).to(bbox.device)

        variances = [0.1, 0.2]
        wh = torch.exp(loc[:, :, 2:] * variances[1]) * default_boxes[:, :, 2:]
        cxcy = loc[:, :, :2] * variances[0] * default_boxes[:, :, 2:] + default_boxes[:, :, :2]
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 2)  # [b, 8732,4]

        batch_size, num_priors, _ = boxes.size()
        boxes = boxes.unsqueeze(2).repeat(1, 1, configer.get('data', 'num_classes'), 1)
        boxes = boxes.contiguous().view(boxes.size(0), -1, 4)

        # clip bounding box
        boxes[:, :, 0::2] = boxes[:, :, 0::2].clamp(min=0, max=input_size[0] - 1)
        boxes[:, :, 1::2] = boxes[:, :, 1::2].clamp(min=0, max=input_size[1] - 1)

        labels = torch.Tensor([i for i in range(configer.get('data', 'num_classes'))]).to(boxes.device)
        labels = labels.view(1, 1, -1, 1).repeat(batch_size, num_priors, 1, 1).contiguous().view(batch_size, -1, 1)
        max_conf = conf.contiguous().view(batch_size, -1, 1)

        # max_conf, labels = conf.max(2, keepdim=True)  # [b, 8732,1]
        predictions = torch.cat((boxes, max_conf.float(), labels.float()), 2)
        output = [None for _ in range(len(predictions))]
        for image_i, image_pred in enumerate(predictions):
            ids = labels[image_i].squeeze(1).nonzero().contiguous().view(-1,)
            if ids.numel() == 0:
                continue

            valid_preds = image_pred[ids]
            _, order = valid_preds[:, 4].sort(0, descending=True)
            order = order[:configer.get('res', 'nms')['pre_nms']]
            valid_preds = valid_preds[order]
            valid_preds = valid_preds[valid_preds[:, 4] > configer.get('res', 'val_conf_thre')]
            if valid_preds.numel() == 0:
                continue

            valid_preds = DetHelper.cls_nms(valid_preds[:, :6],
                                            labels=valid_preds[:, 5],
                                            max_threshold=configer.get('res', 'nms')['max_threshold'],
                                            cls_keep_num=configer.get('res', 'cls_keep_num'))

            _, order = valid_preds[:, 4].sort(0, descending=True)
            order = order[:configer.get('res', 'max_per_image')]
            output[image_i] = valid_preds[order]

        return output

    def __get_info_tree(self, detections, image_raw, input_size):
        height, width, _ = image_raw.shape
        in_width, in_height = input_size
        json_dict = dict()
        object_list = list()
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_pred in detections:
                object_dict = dict()
                xmin = x1.cpu().item() / in_width * width
                ymin = y1.cpu().item() / in_height * height
                xmax = x2.cpu().item() / in_width * width
                ymax = y2.cpu().item() / in_height * height
                object_dict['bbox'] = [xmin, ymin, xmax, ymax]
                object_dict['label'] = int(cls_pred.cpu().item()) - 1
                object_dict['score'] = float('%.2f' % conf.cpu().item())

                object_list.append(object_dict)

        json_dict['objects'] = object_list

        return json_dict

    def debug(self, vis_dir):
        count = 0
        for i, data_dict in enumerate(self.det_data_loader.get_trainloader()):
            inputs = data_dict['img']
            batch_gt_bboxes = data_dict['bboxes']
            batch_gt_labels = data_dict['labels']
            input_size = [inputs.size(3), inputs.size(2)]
            feat_list = list()
            for stride in self.configer.get('network', 'stride_list'):
                feat_list.append(torch.zeros((inputs.size(0), 1, input_size[1] // stride, input_size[0] // stride)))

            bboxes, labels = self.ssd_target_generator(feat_list, batch_gt_bboxes,
                                                       batch_gt_labels, input_size)
            eye_matrix = torch.eye(self.configer.get('data', 'num_classes'))
            labels_target = eye_matrix[labels.view(-1)].view(inputs.size(0), -1,
                                                             self.configer.get('data', 'num_classes'))
            batch_detections = self.decode(bboxes, labels_target,
                                           self.ssd_priorbox_layer(feat_list, input_size), self.configer, input_size)
            for j in range(inputs.size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                ori_img_bgr = self.blob_helper.tensor2bgr(inputs[j])

                self.det_visualizer.vis_default_bboxes(ori_img_bgr,
                                                       self.ssd_priorbox_layer(feat_list, input_size), labels[j])
                json_dict = self.__get_info_tree(batch_detections[j], ori_img_bgr, input_size)
                image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                           json_dict,
                                                           conf_threshold=self.configer.get('res', 'vis_conf_thre'))

                cv2.imwrite(os.path.join(vis_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()

