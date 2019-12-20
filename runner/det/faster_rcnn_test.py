#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


import os
import torch
import torch.nn.functional as F

from data.test.test_data_loader import TestDataLoader
from lib.runner.blob_helper import BlobHelper
from lib.runner.runner_helper import RunnerHelper
from model.det.model_manager import ModelManager
from lib.tools.helper.det_helper import DetHelper
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.helper.json_helper import JsonHelper
from model.det.layers.fr_priorbox_layer import FRPriorBoxLayer
from model.det.layers.fr_roi_generator import FRROIGenerator
from model.det.layers.fr_roi_sampler import FRROISampler
from model.det.layers.rpn_target_assigner import RPNTargetAssigner
from lib.tools.util.logger import Logger as Log
from lib.tools.parser.det_parser import DetParser
from lib.tools.vis.det_visualizer import DetVisualizer
from lib.tools.helper.dc_helper import DCHelper


class FastRCNNTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.det_visualizer = DetVisualizer(configer)
        self.det_parser = DetParser(configer)
        self.det_model_manager = ModelManager(configer)
        self.test_loader = TestDataLoader(configer)
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

    def test(self, test_dir, out_dir):
        for _, data_dict in enumerate(self.test_loader.get_testloader(test_dir=test_dir)):
            data_dict['testing'] = True
            data_dict = RunnerHelper.to_device(self, data_dict)
            out_dict = self.det_net(data_dict)
            meta_list = DCHelper.tolist(data_dict['meta'])
            test_indices_and_rois, test_roi_locs, test_roi_scores, test_rois_num = out_dict['test_group']
            batch_detections = self.decode(test_roi_locs, test_roi_scores, test_indices_and_rois,
                                           test_rois_num, self.configer, meta_list)
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
    def decode(roi_locs, roi_scores, indices_and_rois, test_rois_num, configer, metas):
        indices_and_rois = indices_and_rois
        num_classes = configer.get('data', 'num_classes')
        mean = torch.Tensor(configer.get('roi', 'loc_normalize_mean')).repeat(num_classes)[None]
        std = torch.Tensor(configer.get('roi', 'loc_normalize_std')).repeat(num_classes)[None]
        mean = mean.to(roi_locs.device)
        std = std.to(roi_locs.device)

        roi_locs = (roi_locs * std + mean)
        roi_locs = roi_locs.contiguous().view(-1, num_classes, 4)

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
            tmp_dst_bbox = dst_bbox[start_index:start_index+test_rois_num[i]]
            tmp_dst_bbox[:, :, 0::2] = tmp_dst_bbox[:, :, 0::2].clamp(min=0, max=metas[i]['border_wh'][0] - 1)
            tmp_dst_bbox[:, :, 1::2] = tmp_dst_bbox[:, :, 1::2].clamp(min=0, max=metas[i]['border_wh'][1] - 1)
            tmp_dst_bbox *= (metas[i]['ori_img_size'][0] / metas[i]['border_wh'][0])

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

            valid_ind = DetHelper.cls_nms(valid_preds[:, :5],
                                          labels=valid_preds[:, 5],
                                          max_threshold=configer.get('res', 'nms')['max_threshold'],
                                          return_ind=True)

            valid_preds = valid_preds[valid_ind]
            output[i] = valid_preds

        return output

    def __get_info_tree(self, detections):
        json_dict = dict()
        object_list = list()
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_pred in detections:
                object_dict = dict()
                object_dict['bbox'] = [x1.item(), y1.item(), x2.item(), y2.item()]
                object_dict['label'] = int(cls_pred.cpu().item()) - 1
                object_dict['score'] = float('%.2f' % conf.cpu().item())
                object_list.append(object_dict)

        json_dict['objects'] = object_list

        return json_dict
