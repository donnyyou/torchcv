#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# The class of DenseASPPDetecNet


import torch
from torch import nn
from torch.nn import functional as F

from model.backbone.backbone_selector import BackboneSelector
from model.det.layers.fr_roi_generator import FRROIGenerator
from model.det.layers.fr_roi_sampler import FRROISampler
from model.det.layers.rpn_detection_layer import RPNDetectionLayer
from tools.util.logger import Logger as Log


class FpnRCNN(nn.Module):

    def __init__(self, configer):
        super(FpnRCNN, self).__init__()
        self.configer = configer
        self.backbone = BackboneSelector(configer).get_backbone()
        self.RCNN_layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1,
                                         self.backbone.relu, self.backbone.maxpool)
        self.RCNN_layer1 = nn.Sequential(self.backbone.layer1)
        self.RCNN_layer2 = nn.Sequential(self.backbone.layer2)
        self.RCNN_layer3 = nn.Sequential(self.backbone.layer3)
        self.RCNN_layer4 = nn.Sequential(self.backbone.layer4)
        # Top layer
        self.RCNN_toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # ROI Pool feature downsampling
        self.RCNN_roi_feat_ds = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.RCNN_top = nn.Sequential(
            nn.Conv2d(256, 1024,
                      kernel_size=self.configer.get('roi', 'pooled_height'),
                      stride=self.configer.get('roi', 'pooled_height'), padding=0),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.rpn = NaiveRPN(configer)
        self.roi = FRROIGenerator(configer)
        self.roi_sampler = FRROISampler(configer)
        self.head = RoIHead(configer)

    @staticmethod
    def _upsample_add(x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, *inputs):
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
        input_size = [inputs[0].size(3), inputs[0].size(2)]
        c1 = self.RCNN_layer0(inputs[0])
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        p6 = F.interpolate(p5, scale_factor=0.5, mode='nearest')

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        if self.configer.get('phase') == 'test' and not self.training:
            feat_list, rpn_locs, rpn_scores = self.rpn(rpn_feature_maps)

            indices_and_rois, test_rois_num = self.roi(rpn_locs, rpn_scores,
                                                       self.configer.get('rpn', 'n_test_pre_nms'),
                                                       self.configer.get('rpn', 'n_test_post_nms'))
            roi_cls_locs, roi_scores = self.head(mrcnn_feature_maps, indices_and_rois, input_size)
            return indices_and_rois, roi_cls_locs, roi_scores, test_rois_num

        elif self.configer.get('phase') == 'train' and not self.training:
            _, gt_bboxes, gt_bboxes_num, gt_labels = inputs
            feat_list, rpn_locs, rpn_scores = self.rpn(rpn_feature_maps)
            test_indices_and_rois, test_rois_num = self.roi(feat_list, rpn_locs, rpn_scores,
                                                            self.configer.get('rpn', 'n_test_pre_nms'),
                                                            self.configer.get('rpn', 'n_test_post_nms'),
                                                            input_size=input_size)
            test_roi_cls_locs, test_roi_scores = self.head(mrcnn_feature_maps, test_indices_and_rois, input_size)

            test_group = [test_indices_and_rois, test_roi_cls_locs, test_roi_scores, test_rois_num]
            train_indices_and_rois, _ = self.roi(feat_list, rpn_locs, rpn_scores,
                                                 self.configer.get('rpn', 'n_train_pre_nms'),
                                                 self.configer.get('rpn', 'n_train_post_nms'),
                                                 input_size=input_size)

            sample_rois, gt_roi_bboxes, gt_roi_labels = self.roi_sampler(train_indices_and_rois,
                                                                         gt_bboxes, gt_bboxes_num,
                                                                         gt_labels, input_size)

            sample_roi_locs, sample_roi_scores = self.head(mrcnn_feature_maps, sample_rois, input_size)
            sample_roi_locs = sample_roi_locs.contiguous().view(-1, self.configer.get('data', 'num_classes'), 4)
            sample_roi_locs = sample_roi_locs[
                torch.arange(0, sample_roi_locs.size()[0]).long().to(sample_roi_locs.device),
                gt_roi_labels.long().to(sample_roi_locs.device)].contiguous().view(-1, 4)

            train_group = [rpn_locs, rpn_scores, sample_roi_locs, sample_roi_scores, gt_roi_bboxes, gt_roi_labels]
            return feat_list, train_group, test_group

        elif self.configer.get('phase') == 'train' and self.training:
            _, gt_bboxes, gt_bboxes_num, gt_labels = inputs
            feat_list, rpn_locs, rpn_scores = self.rpn(rpn_feature_maps)

            train_indices_and_rois, _ = self.roi(feat_list, rpn_locs, rpn_scores,
                                                 self.configer.get('rpn', 'n_train_pre_nms'),
                                                 self.configer.get('rpn', 'n_train_post_nms'),
                                                 input_size=input_size)

            sample_rois, gt_roi_bboxes, gt_roi_labels = self.roi_sampler(train_indices_and_rois,
                                                                         gt_bboxes, gt_bboxes_num,
                                                                         gt_labels, input_size)

            sample_roi_locs, sample_roi_scores = self.head(mrcnn_feature_maps, sample_rois, input_size)
            sample_roi_locs = sample_roi_locs.contiguous().view(-1, self.configer.get('data', 'num_classes'), 4)
            sample_roi_locs = sample_roi_locs[
                torch.arange(0, sample_roi_locs.size()[0]).long().to(sample_roi_locs.device),
                gt_roi_labels.long().to(sample_roi_locs.device)].contiguous().view(-1, 4)

            return feat_list, [rpn_locs, rpn_scores, sample_roi_locs, sample_roi_scores, gt_roi_bboxes, gt_roi_labels]

        else:
            Log.error('Invalid Status.')
            exit(1)


class NaiveRPN(nn.Module):
    def __init__(self, configer):
        super(NaiveRPN, self).__init__()
        self.configer = configer
        self.num_anchor_list = self.configer.get('rpn', 'num_anchor_list')
        self.RPN_Conv = nn.Conv2d(256, 512, 3, 1, 1, bias=True)
        self.rpn_detection_layer = RPNDetectionLayer(configer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, rpn_feature_maps):
        feat_list = list()
        for i in range(len(rpn_feature_maps)):
            feat_map = rpn_feature_maps[i]
            rpn_conv = F.relu(self.RPN_Conv(feat_map), inplace=True)
            feat_list.append(rpn_conv)

        rpn_locs, rpn_scores = self.rpn_detection_layer(feat_list)
        return feat_list, rpn_locs, rpn_scores


class RoIHead(nn.Module):
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

    def __init__(self, configer):
        # n_class includes the background
        super(RoIHead, self).__init__()
        self.configer = configer
        self.score = nn.Linear(1024, self.configer.get('data', 'num_classes'))
        if self.configer.get('roi', 'class_agnostic'):
            self.cls_loc = nn.Linear(1024, 4)
        else:
            self.cls_loc = nn.Linear(1024, 4 * self.configer.get('data', 'num_classes'))

        from exts.roialign.module import RoIAlign2D
        self.roi_align = RoIAlign2D(pooled_height=int(self.configer.get('roi', 'pooled_height')),
                                    pooled_width=int(self.configer.get('roi', 'pooled_width')),
                                    spatial_scale=1.0 / float(self.configer.get('roi', 'spatial_stride')),
                                    sampling_ratio=2)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

    def forward(self, feat_maps, indices_and_rois, input_size):
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
        h = indices_and_rois.data[:, 4] - indices_and_rois.data[:, 2] + 1
        w = indices_and_rois.data[:, 3] - indices_and_rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5

        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(2, 6)):
            if (roi_level == l).sum() == 0:
                continue

            idx_l = (roi_level == l).nonzero().squeeze()
            box_to_levels.append(idx_l)
            scale = feat_maps[i].size(2) / input_size[1]
            feat = self.roi_align(feat_maps[i], indices_and_rois[idx_l], scale)
            roi_pool_feats.append(feat)

        roi_pool_feat = torch.cat(roi_pool_feats, 0)
        box_to_level = torch.cat(box_to_levels, 0)
        idx_sorted, order = torch.sort(box_to_level)
        roi_pool_feat = roi_pool_feat[order]

        block5 = self.RCNN_top(roi_pool_feat)
        pooled_feat = block5.mean(3).mean(2)
        roi_cls_locs = self.cls_loc(pooled_feat)
        roi_scores = self.score(pooled_feat)
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
