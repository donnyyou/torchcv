#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


import os
import cv2
import torch

from datasets.det.data_loader import DataLoader
from methods.tools.blob_helper import BlobHelper
from methods.tools.runner_helper import RunnerHelper
from models.det.model_manager import ModelManager
from utils.helpers.det_helper import DetHelper
from utils.helpers.image_helper import ImageHelper
from utils.helpers.json_helper import JsonHelper
from models.det.layers.yolo_detection_layer import YOLODetectionLayer
from models.det.layers.yolo_target_generator import YOLOTargetGenerator
from utils.tools.logger import Logger as Log
from utils.parser.det_parser import DetParser
from utils.visualizer.det_visualizer import DetVisualizer


class YOLOv3Test(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.det_visualizer = DetVisualizer(configer)
        self.det_parser = DetParser(configer)
        self.det_model_manager = ModelManager(configer)
        self.det_data_loader = DataLoader(configer)
        self.yolo_target_generator = YOLOTargetGenerator(configer)
        self.yolo_detection_layer = YOLODetectionLayer(configer)
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
                                             input_size=self.configer.get('data', 'input_size'), scale=1.0)

        with torch.no_grad():
            inputs = inputs.unsqueeze(0).to(self.device)
            _, _, detections = self.det_net(inputs)

        batch_detections = self.decode(detections, self.configer)
        json_dict = self.__get_info_tree(batch_detections[0], ori_img_bgr)

        image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                   json_dict,
                                                   conf_threshold=self.configer.get('res', 'vis_conf_thre'))
        ImageHelper.save(ori_img_bgr, raw_path)
        ImageHelper.save(image_canvas, vis_path)

        Log.info('Json Path: {}'.format(json_path))
        JsonHelper.save_file(json_dict, json_path)
        return json_dict

    @staticmethod
    def decode(batch_pred_bboxes, configer, input_size):
        box_corner = batch_pred_bboxes.new(batch_pred_bboxes.shape)
        box_corner[:, :, 0] = batch_pred_bboxes[:, :, 0] - batch_pred_bboxes[:, :, 2] / 2
        box_corner[:, :, 1] = batch_pred_bboxes[:, :, 1] - batch_pred_bboxes[:, :, 3] / 2
        box_corner[:, :, 2] = batch_pred_bboxes[:, :, 0] + batch_pred_bboxes[:, :, 2] / 2
        box_corner[:, :, 3] = batch_pred_bboxes[:, :, 1] + batch_pred_bboxes[:, :, 3] / 2

        # clip bounding box
        box_corner[:, :, 0::2] = box_corner[:, :, 0::2].clamp(min=0, max=1.0)
        box_corner[:, :, 1::2] = box_corner[:, :, 1::2].clamp(min=0, max=1.0)

        batch_pred_bboxes[:, :, :4] = box_corner[:, :, :4]
        batch_pred_bboxes[:, :, 0::2] *= input_size[0]
        batch_pred_bboxes[:, :, 1::2] *= input_size[1]
        output = [None for _ in range(len(batch_pred_bboxes))]
        for image_i, image_pred in enumerate(batch_pred_bboxes):
            # Filter out confidence scores below threshold
            conf_mask = (image_pred[:, 4] > configer.get('res', 'val_conf_thre')).squeeze()
            image_pred = image_pred[conf_mask]
            # If none are remaining => process next image
            if image_pred.numel() == 0:
                continue

            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5:5 + configer.get('data', 'num_classes')], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            output[image_i] = DetHelper.cls_nms(detections,
                                                labels=class_pred.squeeze(1),
                                                max_threshold=configer.get('res', 'nms')['max_threshold'])

        return output

    def __get_info_tree(self, detections, image_raw, input_size):
        height, width, _ = image_raw.shape
        json_dict = dict()
        object_list = list()
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                object_dict = dict()
                xmin = x1.cpu().item() / input_size[0] * width
                ymin = y1.cpu().item() / input_size[1] * height
                xmax = x2.cpu().item() / input_size[0] * width
                ymax = y2.cpu().item() / input_size[1] * height
                object_dict['bbox'] = [xmin, ymin, xmax, ymax]
                object_dict['label'] = int(cls_pred.cpu().item())
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

            targets, _, _ = self.yolo_target_generator(feat_list, batch_gt_bboxes,
                                                       batch_gt_labels, input_size)
            targets = targets.to(self.device)
            anchors_list = self.configer.get('gt', 'anchors_list')
            output_list = list()
            be_c = 0
            for f_index, anchors in enumerate(anchors_list):
                feat_stride = self.configer.get('network', 'stride_list')[f_index]
                fm_size = [int(round(border / feat_stride)) for border in input_size]
                num_c = len(anchors) * fm_size[0] * fm_size[1]
                output_list.append(targets[:, be_c:be_c+num_c].contiguous()
                                   .view(targets.size(0), len(anchors), fm_size[1], fm_size[0], -1)
                                   .permute(0, 1, 4, 2, 3).contiguous()
                                   .view(targets.size(0), -1, fm_size[1], fm_size[0]))

                be_c += num_c

            batch_detections = self.decode(self.yolo_detection_layer(output_list)[2], self.configer, input_size)

            for j in range(inputs.size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                ori_img_bgr = self.blob_helper.tensor2bgr(inputs[j])

                json_dict = self.__get_info_tree(batch_detections[j], ori_img_bgr, input_size)

                image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                           json_dict,
                                                           conf_threshold=self.configer.get('vis', 'obj_threshold'))

                cv2.imwrite(os.path.join(vis_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()
