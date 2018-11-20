#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (donnyyou@163.com)
# Class Definition for Single Shot Detector.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import torch
import torch.nn.functional as F

from datasets.det_data_loader import DetDataLoader
from datasets.tools.data_transformer import DataTransformer
from datasets.tools.transforms import BoundResize
from methods.tools.blob_helper import BlobHelper
from methods.tools.module_utilizer import ModuleUtilizer
from models.det_model_manager import DetModelManager
from utils.helpers.det_helper import DetHelper
from utils.helpers.file_helper import FileHelper
from utils.helpers.image_helper import ImageHelper
from utils.helpers.json_helper import JsonHelper
from utils.layers.det.fr_priorbox_layer import FRPriorBoxLayer
from utils.layers.det.fr_roi_generator import FRRoiGenerator
from utils.layers.det.fr_roi_sample_layer import FRRoiSampleLayer
from utils.layers.det.rpn_target_generator import RPNTargetGenerator
from utils.tools.logger import Logger as Log
from vis.parser.det_parser import DetParser
from vis.visualizer.det_visualizer import DetVisualizer


class FastRCNNTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.det_visualizer = DetVisualizer(configer)
        self.det_parser = DetParser(configer)
        self.det_model_manager = DetModelManager(configer)
        self.det_data_loader = DetDataLoader(configer)
        self.roi_sampler = FRRoiSampleLayer(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.rpn_target_generator = RPNTargetGenerator(configer)
        self.fr_priorbox_layer = FRPriorBoxLayer(configer)
        self.fr_roi_generator = FRRoiGenerator(configer)
        self.data_transformer = DataTransformer(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.det_net = None

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = self.module_utilizer.load_net(self.det_net)
        self.det_net.eval()

    def __test_img(self, image_path, json_path, raw_path, vis_path):
        Log.info('Image Path: {}'.format(image_path))
        img = ImageHelper.read_image(image_path,
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        ori_img_bgr = ImageHelper.get_cv2_bgr(img, mode=self.configer.get('data', 'input_mode'))
        img, scale = BoundResize()(img)
        inputs = self.blob_helper.make_input(img, scale=1.0)
        with torch.no_grad():
            # Forward pass.
            test_group = self.det_net(inputs, scale)

            test_indices_and_rois, test_roi_locs, test_roi_scores, test_rois_num = test_group

        batch_detections = self.decode(test_roi_locs,
                                       test_roi_scores,
                                       test_indices_and_rois,
                                       test_rois_num,
                                       self.configer,
                                       ImageHelper.get_size(img))
        json_dict = self.__get_info_tree(batch_detections[0], ori_img_bgr, scale=scale)

        image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                   json_dict,
                                                   conf_threshold=self.configer.get('vis', 'conf_threshold'))
        cv2.imwrite(vis_path, image_canvas)
        cv2.imwrite(raw_path, ori_img_bgr)

        Log.info('Json Path: {}'.format(json_path))
        JsonHelper.save_file(json_dict, json_path)
        return json_dict

    @staticmethod
    def decode(roi_locs, roi_scores, indices_and_rois, test_rois_num, configer, input_size):
        roi_locs = roi_locs.cpu()
        roi_scores = roi_scores.cpu()
        indices_and_rois = indices_and_rois.cpu()
        num_classes = configer.get('data', 'num_classes')
        mean = torch.Tensor(configer.get('roi', 'loc_normalize_mean')).repeat(num_classes)[None]
        std = torch.Tensor(configer.get('roi', 'loc_normalize_std')).repeat(num_classes)[None]
        mean = mean.to(roi_locs.device)
        std = std.to(roi_locs.device)

        roi_locs = (roi_locs * std + mean)
        roi_locs = roi_locs.contiguous().view(-1, num_classes, 4)
        # roi_locs = roi_locs[:,:, [1, 0, 3, 2]]

        rois = indices_and_rois[:, 1:]
        rois = rois.contiguous().view(-1, 1, 4).expand_as(roi_locs)
        wh = torch.exp(roi_locs[:, :, 2:]) * (rois[:, :, 2:] - rois[:, :, :2])
        cxcy = roi_locs[:, :, :2] * (rois[:, :, 2:] - rois[:, :, :2]) + (rois[:, :, :2] + rois[:, :, 2:]) / 2
        dst_bbox = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 2)  # [b, 8732,4]

        # clip bounding box
        dst_bbox[:, :, 0::2] = (dst_bbox[:, :, 0::2]).clamp(min=0, max=input_size[0]-1)
        dst_bbox[:, :, 1::2] = (dst_bbox[:, :, 1::2]).clamp(min=0, max=input_size[1]-1)

        if configer.get('phase') != 'debug':
            cls_prob = F.softmax(roi_scores, dim=1)
        else:
            cls_prob = roi_scores

        cls_label = torch.LongTensor([i for i in range(num_classes)])\
            .contiguous().view(1, num_classes).repeat(indices_and_rois.size(0), 1)

        output = [None for _ in range(test_rois_num.size(0))]
        start_index = 0
        for i in range(test_rois_num.size(0)):
            # batch_index = (indices_and_rois[:, 0] == i).nonzero().contiguous().view(-1,)
            # tmp_dst_bbox = dst_bbox[batch_index]
            # tmp_cls_prob = cls_prob[batch_index]
            # tmp_cls_label = cls_label[batch_index]
            tmp_dst_bbox = dst_bbox[start_index:start_index+test_rois_num[i]]
            tmp_cls_prob = cls_prob[start_index:start_index+test_rois_num[i]]
            tmp_cls_label = cls_label[start_index:start_index+test_rois_num[i]]
            start_index += test_rois_num[i]

            mask = (tmp_cls_prob > configer.get('vis', 'conf_threshold')) & (tmp_cls_label > 0)

            tmp_dst_bbox = tmp_dst_bbox[mask].contiguous().view(-1, 4)
            if tmp_dst_bbox.numel() == 0:
                continue

            tmp_cls_prob = tmp_cls_prob[mask].contiguous().view(-1,).unsqueeze(1)
            tmp_cls_label = tmp_cls_label[mask].contiguous().view(-1,).unsqueeze(1)

            valid_preds = torch.cat((tmp_dst_bbox, tmp_cls_prob.float(), tmp_cls_label.float()), 1)

            keep = DetHelper.cls_nms(valid_preds[:, :4],
                                     scores=valid_preds[:, 4],
                                     labels=valid_preds[:, 5],
                                     nms_threshold=configer.get('nms', 'overlap_threshold'),
                                     iou_mode=configer.get('nms', 'mode'))

            output[i] = valid_preds[keep]

        return output

    def __make_tensor(self, gt_bboxes, gt_labels):
        len_arr = [gt_labels[i].numel() for i in range(len(gt_bboxes))]
        batch_maxlen = max(max(len_arr), 1)
        target_bboxes = torch.zeros((len(gt_bboxes), batch_maxlen, 4)).float()
        target_labels = torch.zeros((len(gt_bboxes), batch_maxlen)).long()
        for i in range(len(gt_bboxes)):
            target_bboxes[i, :len_arr[i], :] = gt_bboxes[i]
            target_labels[i, :len_arr[i]] = gt_labels[i]

        target_bboxes_num = torch.Tensor(len_arr).long()
        return target_bboxes, target_bboxes_num, target_labels

    def __get_info_tree(self, detections, image_raw, scale=1.0):
        height, width, _ = image_raw.shape
        json_dict = dict()
        object_list = list()
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_pred in detections:
                object_dict = dict()
                xmin = min(x1.cpu().item() / scale, width - 1)
                ymin = min(y1.cpu().item() / scale, height - 1)
                xmax = min(x2.cpu().item() / scale, width - 1)
                ymax = min(y2.cpu().item() / scale, height - 1)
                object_dict['bbox'] = [xmin, ymin, xmax, ymax]
                object_dict['label'] = int(cls_pred.cpu().item()) - 1
                object_dict['score'] = float('%.2f' % conf.cpu().item())

                object_list.append(object_dict)

        json_dict['objects'] = object_list

        return json_dict

    def test(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/det', self.configer.get('dataset'))

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
                                'vis/results/det', self.configer.get('dataset'), 'debug')

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        count = 0
        for i, data_dict in enumerate(self.det_data_loader.get_trainloader()):
            img_scale = data_dict['imgscale']
            inputs = data_dict['img']
            batch_gt_bboxes = data_dict['bboxes']
            # batch_gt_bboxes = ResizeBoxes()(inputs, data_dict['bboxes'])
            batch_gt_labels = data_dict['labels']

            input_size = [inputs.size(3), inputs.size(2)]
            feat_list = list()
            for stride in self.configer.get('rpn', 'stride_list'):
                feat_list.append(torch.zeros((inputs.size(0), 1, input_size[1] // stride, input_size[0] // stride)))

            gt_rpn_locs, gt_rpn_labels = self.rpn_target_generator(feat_list, batch_gt_bboxes, input_size)
            eye_matrix = torch.eye(2)
            gt_rpn_labels[gt_rpn_labels == -1] = 0
            gt_rpn_scores = eye_matrix[gt_rpn_labels.view(-1)].view(inputs.size(0), -1, 2)
            test_indices_and_rois, _ = self.fr_roi_generator(feat_list, gt_rpn_locs, gt_rpn_scores,
                                                             self.configer.get('rpn', 'n_test_pre_nms'),
                                                             self.configer.get('rpn', 'n_test_post_nms'),
                                                             input_size, img_scale)

            gt_bboxes, gt_nums, gt_labels = self.__make_tensor(batch_gt_bboxes, batch_gt_labels)
            sample_rois, gt_roi_locs, gt_roi_labels = self.roi_sampler(test_indices_and_rois,
                                                                       gt_bboxes, gt_nums, gt_labels, input_size)

            self.det_visualizer.vis_rois(inputs, sample_rois[gt_roi_labels > 0])
            gt_cls_roi_locs = torch.zeros((gt_roi_locs.size(0), self.configer.get('data', 'num_classes'), 4))
            gt_cls_roi_locs[torch.arange(0, sample_rois.size(0)).long(), gt_roi_labels.long()] = gt_roi_locs
            gt_cls_roi_locs = gt_cls_roi_locs.contiguous().view(-1, 4*self.configer.get('data', 'num_classes'))
            eye_matrix = torch.eye(self.configer.get('data', 'num_classes'))

            gt_roi_scores = eye_matrix[gt_roi_labels.view(-1)].view(gt_roi_labels.size(0),
                                                                    self.configer.get('data', 'num_classes'))
            test_rois_num = torch.zeros((len(gt_bboxes), )).long()
            for batch_id in range(len(gt_bboxes)):
                batch_index = (sample_rois[:, 0] == batch_id).nonzero().contiguous().view(-1,)
                test_rois_num[batch_id] = batch_index.numel()

            batch_detections = FastRCNNTest.decode(gt_cls_roi_locs, gt_roi_scores,
                                                   sample_rois, test_rois_num, self.configer, input_size)

            for j in range(inputs.size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                ori_img_bgr = self.blob_helper.tensor2bgr(inputs[j])

                self.det_visualizer.vis_default_bboxes(ori_img_bgr,
                                                       self.fr_priorbox_layer(feat_list, input_size),
                                                       gt_rpn_labels[j])
                json_dict = self.__get_info_tree(batch_detections[j], ori_img_bgr)
                image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                           json_dict,
                                                           conf_threshold=self.configer.get('vis', 'conf_threshold'))

                cv2.imwrite(os.path.join(base_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()
