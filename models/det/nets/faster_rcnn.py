#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16

from models.det.loss.det_modules import FRLoss
from models.det.layers.fr_roi_generator import FRROIGenerator
from models.det.layers.fr_roi_sampler import FRROISampler
from models.det.layers.rpn_detection_layer import RPNDetectionLayer
from models.det.layers.rpn_target_assigner import RPNTargetAssigner
from utils.tools.logger import Logger as Log

try:
    from extensions.ops.roi_pool.modules.roi_pool import RoIPool
except ImportError:
    print('RoIPool ImportError.')


DETECTOR_CONFIG = {
    'vgg_cfg': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
}


class VGGModel(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self):
        # the 30th layer of features is relu of conv5_3
        model = vgg16(pretrained=False)
        if self.configer.get('network', 'pretrained') is not None :
            Log.info('Loading pretrained model: {}'.format(self.configer.get('network', 'pretrained')))
            model.load_state_dict(torch.load(self.configer.get('network', 'pretrained')))

        features = list(model.features)[:30]
        classifier = model.classifier

        classifier = list(classifier)
        del classifier[6]
        if not self.configer.get('network', 'use_drop'):
            del classifier[5]
            del classifier[2]

        classifier = nn.Sequential(*classifier)

        # freeze top4 conv
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        return nn.Sequential(*features), classifier


class FasterRCNN(nn.Module):

    def __init__(self, configer):
        super(FasterRCNN, self).__init__()
        self.configer = configer
        self.backbone, self.classifier = VGGModel(configer)()
        self.rpn = NaiveRPN(configer)
        self.rpn_target_assigner = RPNTargetAssigner(configer)
        self.roi_generator = FRROIGenerator(configer)
        self.roi_sampler = FRROISampler(configer)
        self.bbox_head = BBoxHead(configer, self.classifier)
        self.det_loss = FRLoss(self.configer)

    def forward(self, data_dict):
        """Forward Faster R-CNN.
        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.
        Here are notations used.
        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.
        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.
        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.
        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.
            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.
        """
        if self.configer.get('phase') == 'test' and not self.training:
            x = self.backbone(data_dict['img'])
            feat_list, rpn_locs, rpn_scores = self.rpn(x)
            indices_and_rois, test_rois_num = self.roi_generator(feat_list, rpn_locs, rpn_scores,
                                                                 self.configer.get('rpn', 'n_test_pre_nms'),
                                                                 self.configer.get('rpn', 'n_test_post_nms'),
                                                                 data_dict['meta'])
            roi_cls_locs, roi_scores = self.bbox_head(x, indices_and_rois, data_dict['meta'])
            return indices_and_rois, roi_cls_locs, roi_scores, test_rois_num

        elif self.configer.get('phase') == 'train' and not self.training:
            x = self.backbone(data_dict['img'])
            feat_list, rpn_locs, rpn_scores = self.rpn(x)
            gt_rpn_locs, gt_rpn_labels = self.rpn_target_assigner(feat_list, data_dict['bboxes'], data_dict['meta'])

            test_indices_and_rois, test_rois_num = self.roi_generator(feat_list, rpn_locs, rpn_scores,
                                                                      self.configer.get('rpn', 'n_test_pre_nms'),
                                                                      self.configer.get('rpn', 'n_test_post_nms'),
                                                                      data_dict['meta'])
            test_roi_cls_locs, test_roi_scores = self.bbox_head(x, test_indices_and_rois, data_dict['meta'])
            test_group = [test_indices_and_rois, test_roi_cls_locs, test_roi_scores, test_rois_num]
            train_indices_and_rois, _ = self.roi_generator(feat_list, rpn_locs, rpn_scores,
                                                           self.configer.get('rpn', 'n_train_pre_nms'),
                                                           self.configer.get('rpn', 'n_train_post_nms'),
                                                           data_dict['meta'])
            sample_rois, gt_roi_bboxes, gt_roi_labels = self.roi_sampler(train_indices_and_rois,
                                                                         data_dict['bboxes'],
                                                                         data_dict['labels'],
                                                                         data_dict['meta'])

            sample_roi_locs, sample_roi_scores = self.bbox_head(x, sample_rois, data_dict['meta'])
            sample_roi_locs = sample_roi_locs.contiguous().view(-1, self.configer.get('data', 'num_classes'), 4)
            sample_roi_locs = sample_roi_locs[
                torch.arange(0, sample_roi_locs.size()[0]).long().to(sample_roi_locs.device),
                gt_roi_labels.long().to(sample_roi_locs.device)].contiguous().view(-1, 4)
            train_group = [rpn_locs, rpn_scores, sample_roi_locs, sample_roi_scores]
            target_group = [gt_rpn_locs, gt_rpn_labels, gt_roi_bboxes, gt_roi_labels]
            return self.det_loss(train_group, target_group), test_group

        elif self.configer.get('phase') == 'train' and self.training:
            x = self.backbone(data_dict['img'])
            feat_list, rpn_locs, rpn_scores = self.rpn(x)
            gt_rpn_locs, gt_rpn_labels = self.rpn_target_assigner(feat_list, data_dict['bboxes'], data_dict['meta'])
            train_indices_and_rois, _ = self.roi_generator(feat_list, rpn_locs, rpn_scores,
                                                           self.configer.get('rpn', 'n_train_pre_nms'),
                                                           self.configer.get('rpn', 'n_train_post_nms'),
                                                           data_dict['meta'])
            sample_rois, gt_roi_bboxes, gt_roi_labels = self.roi_sampler(train_indices_and_rois,
                                                                         data_dict['bboxes'],
                                                                         data_dict['labels'],
                                                                         data_dict['meta'])
            sample_roi_locs, sample_roi_scores = self.bbox_head(x, sample_rois, data_dict['meta'])
            sample_roi_locs = sample_roi_locs.contiguous().view(-1, self.configer.get('data', 'num_classes'), 4)
            sample_roi_locs = sample_roi_locs[
                torch.arange(0, sample_roi_locs.size()[0]).long().to(sample_roi_locs.device),
                gt_roi_labels.long().to(sample_roi_locs.device)].contiguous().view(-1, 4)
            train_group = [rpn_locs, rpn_scores, sample_roi_locs, sample_roi_scores]
            target_group = [gt_rpn_locs, gt_rpn_labels, gt_roi_bboxes, gt_roi_labels]
            return self.det_loss(train_group, target_group)

        else:
            Log.error('Invalid Status.')
            exit(1)


class NaiveRPN(nn.Module):
    def __init__(self, configer):
        super(NaiveRPN, self).__init__()
        self.configer = configer
        self.num_anchor_list = self.configer.get('rpn', 'num_anchor_list')
        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.rpn_detection_layer = RPNDetectionLayer(configer)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        h = F.relu(self.conv1(x))
        rpn_locs, rpn_scores = self.rpn_detection_layer([h])
        return [h], rpn_locs, rpn_scores


class BBoxHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16
    """

    def __init__(self, configer, classifier):
        # n_class includes the background
        super(BBoxHead, self).__init__()
        self.configer = configer
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, self.configer.get('data', 'num_classes') * 4)
        self.score = nn.Linear(4096, self.configer.get('data', 'num_classes'))
        self.roi_pool = RoIPool(out_size=tuple(self.configer.get('roi', 'pooled_hw')),
                                spatial_scale=1.0 / float(self.configer.get('roi', 'spatial_stride')))

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

    def forward(self, x, indices_and_rois, meta):
        """Forward the chain.
        We assume that there are :math:`N` batches.
        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
        """
        # in case roi_indices is  ndarray
        pool = self.roi_pool(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        if m.bias is not None:
            m.bias.data.zero_()
