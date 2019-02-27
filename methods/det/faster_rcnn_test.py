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
from models.det.layers.fr_priorbox_layer import FRPriorBoxLayer
from models.det.layers.fr_roi_generator import FRROIGenerator
from models.det.layers.fr_roi_sampler import FRROISampler
from models.det.layers.rpn_target_assigner import RPNTargetAssigner
from utils.tools.logger import Logger as Log
from utils.parser.det_parser import DetParser
from utils.visualizer.det_visualizer import DetVisualizer
from extensions.tools.parallel import DataContainer
from utils.helpers.dc_helper import DCHelper


class FastRCNNTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.det_visualizer = DetVisualizer(configer)
        self.det_parser = DetParser(configer)
        self.det_model_manager = ModelManager(configer)
        self.det_data_loader = DataLoader(configer)
        self.roi_sampler = FRROISampler(configer)
        self.rpn_target_generator = RPNTargetAssigner(configer)
        self.fr_priorbox_layer = FRPriorBoxLayer(configer)
        self.fr_roi_generator = FRROIGenerator(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.det_net = None

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = RunnerHelper.load_net(self, self.det_net)
        self.det_net.eval()

    def __test_img(self, image_path, json_path, raw_path, vis_path):
        Log.info('Image Path: {}'.format(image_path))
        image = ImageHelper.read_image(image_path,
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        ori_img_bgr = ImageHelper.get_cv2_bgr(image, mode=self.configer.get('data', 'input_mode'))
        width, height = ImageHelper.get_size(image)
        scale1 = self.configer.get('test', 'resize_bound')[0] / min(width, height)
        scale2 = self.configer.get('test', 'resize_bound')[1] / max(width, height)
        scale = min(scale1, scale2)
        inputs = self.blob_helper.make_input(image, scale=scale)
        b, c, h, w = inputs.size()
        border_wh = [w, h]
        if self.configer.exists('test', 'fit_stride'):
            stride = self.configer.get('test', 'fit_stride')

            pad_w = 0 if (w % stride == 0) else stride - (w % stride)  # right
            pad_h = 0 if (h % stride == 0) else stride - (h % stride)  # down

            expand_image = torch.zeros((b, c, h + pad_h, w + pad_w)).to(inputs.device)
            expand_image[:, :, 0:h, 0:w] = inputs
            inputs= expand_image

        data_dict = dict(
            img=inputs,
            meta=DataContainer([[dict(ori_img_size=ImageHelper.get_size(ori_img_bgr),
                                      aug_img_size=border_wh,
                                      img_scale=scale,
                                      input_size=[inputs.size(3), inputs.size(2)])]], cpu_only=True)
        )

        with torch.no_grad():
            # Forward pass.
            test_group = self.det_net(data_dict)

            test_indices_and_rois, test_roi_locs, test_roi_scores, test_rois_num = test_group

        batch_detections = self.decode(test_roi_locs,
                                       test_roi_scores,
                                       test_indices_and_rois,
                                       test_rois_num,
                                       self.configer,
                                       DCHelper.tolist(data_dict['meta']))
        json_dict = self.__get_info_tree(batch_detections[0], ori_img_bgr, scale=scale)

        image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                   json_dict,
                                                   conf_threshold=self.configer.get('res', 'vis_conf_thre'))
        cv2.imwrite(vis_path, image_canvas)
        cv2.imwrite(raw_path, ori_img_bgr)

        Log.info('Json Path: {}'.format(json_path))
        JsonHelper.save_file(json_dict, json_path)
        return json_dict

    @staticmethod
    def decode(roi_locs, roi_scores, indices_and_rois, test_rois_num, configer, metas):
        indices_and_rois = indices_and_rois
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

        if configer.get('phase') != 'debug':
            cls_prob = F.softmax(roi_scores, dim=1)
        else:
            cls_prob = roi_scores

        cls_label = torch.LongTensor([i for i in range(num_classes)])\
            .contiguous().view(1, num_classes).repeat(indices_and_rois.size(0), 1).to(roi_locs.device)

        output = [None for _ in range(test_rois_num.size(0))]
        start_index = 0
        for i in range(test_rois_num.size(0)):
            # batch_index = (indices_and_rois[:, 0] == i).nonzero().contiguous().view(-1,)
            # tmp_dst_bbox = dst_bbox[batch_index]
            # tmp_cls_prob = cls_prob[batch_index]
            # tmp_cls_label = cls_label[batch_index]
            tmp_dst_bbox = dst_bbox[start_index:start_index+test_rois_num[i]]
            # clip bounding box
            tmp_dst_bbox[:, :, 0::2] = tmp_dst_bbox[:, :, 0::2].clamp(min=0, max=metas[i]['border_size'][0] - 1)
            tmp_dst_bbox[:, :, 1::2] = tmp_dst_bbox[:, :, 1::2].clamp(min=0, max=metas[i]['border_size'][1] - 1)

            tmp_cls_prob = cls_prob[start_index:start_index+test_rois_num[i]]
            tmp_cls_label = cls_label[start_index:start_index+test_rois_num[i]]
            start_index += test_rois_num[i]

            mask = (tmp_cls_prob > configer.get('res', 'val_conf_thre')) & (tmp_cls_label > 0)

            tmp_dst_bbox = tmp_dst_bbox[mask].contiguous().view(-1, 4)
            if tmp_dst_bbox.numel() == 0:
                continue

            tmp_cls_prob = tmp_cls_prob[mask].contiguous().view(-1,).unsqueeze(1)
            tmp_cls_label = tmp_cls_label[mask].contiguous().view(-1,).unsqueeze(1)

            valid_preds = torch.cat((tmp_dst_bbox, tmp_cls_prob.float(), tmp_cls_label.float()), 1)

            output[i] = DetHelper.cls_nms(valid_preds,
                                          labels=valid_preds[:, 5],
                                          max_threshold=configer.get('res', 'nms')['max_threshold'])

        return output

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

    def debug(self, vis_dir):
        count = 0
        for i, data_dict in enumerate(self.det_data_loader.get_trainloader()):
            feat_list = list()
            input_size = data_dict['meta'][0]['input_size']
            for stride in self.configer.get('rpn', 'stride_list'):
                feat_list.append(torch.zeros((data_dict['img'].size(0), 1,
                                              input_size[1] // stride, input_size[0] // stride)))

            gt_rpn_locs, gt_rpn_labels = self.rpn_target_generator(feat_list, data_dict['bboxes'], data_dict['meta'])
            eye_matrix = torch.eye(2)
            gt_rpn_labels[gt_rpn_labels == -1] = 0
            gt_rpn_scores = eye_matrix[gt_rpn_labels.view(-1)].view(data_dict['img'].size(0), -1, 2)
            test_indices_and_rois, _ = self.fr_roi_generator(feat_list, gt_rpn_locs, gt_rpn_scores,
                                                             self.configer.get('rpn', 'n_test_pre_nms'),
                                                             self.configer.get('rpn', 'n_test_post_nms'),
                                                             data_dict['meta'])

            sample_rois, gt_roi_locs, gt_roi_labels = self.roi_sampler(test_indices_and_rois,
                                                                       data_dict['bboxes'],
                                                                       data_dict['labels'],
                                                                       data_dict['meta'])

            self.det_visualizer.vis_rois(data_dict['img'], sample_rois[gt_roi_labels > 0])
            gt_cls_roi_locs = torch.zeros((gt_roi_locs.size(0), self.configer.get('data', 'num_classes'), 4))
            gt_cls_roi_locs[torch.arange(0, sample_rois.size(0)).long(), gt_roi_labels.long()] = gt_roi_locs
            gt_cls_roi_locs = gt_cls_roi_locs.contiguous().view(-1, 4*self.configer.get('data', 'num_classes'))
            eye_matrix = torch.eye(self.configer.get('data', 'num_classes'))

            gt_roi_scores = eye_matrix[gt_roi_labels.view(-1)].view(gt_roi_labels.size(0),
                                                                    self.configer.get('data', 'num_classes'))
            test_rois_num = torch.zeros((len(data_dict['bboxes']), )).long()
            for batch_id in range(len(data_dict['bboxes'])):
                batch_index = (sample_rois[:, 0] == batch_id).nonzero().contiguous().view(-1,)
                test_rois_num[batch_id] = batch_index.numel()

            batch_detections = FastRCNNTest.decode(gt_cls_roi_locs, gt_roi_scores,
                                                   sample_rois, test_rois_num, self.configer, data_dict['meta'])

            for j in range(data_dict['img'].size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                ori_img_bgr = self.blob_helper.tensor2bgr(data_dict['img'][j])

                self.det_visualizer.vis_default_bboxes(ori_img_bgr,
                                                       self.fr_priorbox_layer(feat_list, input_size),
                                                       gt_rpn_labels[j])
                json_dict = self.__get_info_tree(batch_detections[j], ori_img_bgr)
                image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                           json_dict,
                                                           conf_threshold=self.configer.get('res', 'vis_conf_thre'))

                cv2.imwrite(os.path.join(vis_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()
