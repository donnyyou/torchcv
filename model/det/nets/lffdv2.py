# -*- coding: utf-8 -*-
"""
network description:
network structure:
"""
import torch
import torch.nn as nn

from model.det.layers.ssd_priorbox_layer import SSDPriorBoxLayer
from model.det.layers.ssd_target_generator import SSDTargetGenerator
from model.det.loss.loss import BASE_LOSS_DICT
from lib.tools.util.logger import Logger as Log

num_filters_list = [32, 64, 128, 256]


class DetectionLayer(nn.Module):

    def __init__(self, configer):
        super(DetectionLayer, self).__init__()
        self.ssd_priorbox_layer = SSDPriorBoxLayer(configer)
        self.num_classes = configer.get('data', 'num_classes')
        self.num_anchors = configer.get('anchor', 'num_anchor_list')
        self.num_features = configer.get('network', 'num_feature_list')
        self.ssd_head_index = configer.get('network', 'head_index_list')

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        self.pre_conv = nn.ModuleList()

        for i in range(max(self.ssd_head_index) + 1):
            self.pre_conv.append(
                nn.Sequential(
                    nn.Conv2d(self.num_features[i], num_filters_list[2], 1),
                    nn.ReLU()
                )
            )
            self.loc_layers.append(
                nn.Sequential(
                    nn.Conv2d(num_filters_list[2], num_filters_list[2], 1),
                    nn.ReLU(),
                    nn.Conv2d(num_filters_list[2], self.num_anchors[i] * 4, kernel_size=1, padding=0)
                )
            )
            self.conf_layers.append(
                nn.Sequential(
                    nn.Conv2d(num_filters_list[2], num_filters_list[2], 1),
                    nn.ReLU(),
                    nn.Conv2d(num_filters_list[2], self.num_anchors[i] * self.num_classes, kernel_size=1, padding=0)
                )
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat_list, data_dict):
        y_locs = []
        y_confs = []

        for i, x in enumerate(feat_list):
            x = self.pre_conv[self.ssd_head_index[i]](x)
            y_loc = self.loc_layers[self.ssd_head_index[i]](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[self.ssd_head_index[i]](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)
            y_confs.append(y_conf)

        pred_loc = torch.cat(y_locs, 1)
        pred_conf = torch.cat(y_confs, 1)
        input_size = [data_dict['img'].size(3), data_dict['img'].size(2)]
        default_boxes = self.ssd_priorbox_layer(feat_list, input_size)
        default_boxes = default_boxes.unsqueeze(0).repeat(pred_loc.size(0), 1, 1).to(pred_loc.device)

        variances = [0.1, 0.2]
        wh = torch.exp(pred_loc[:, :, 2:] * variances[1]) * default_boxes[:, :, 2:]
        cxcy = pred_loc[:, :, :2] * variances[0] * default_boxes[:, :, 2:] + default_boxes[:, :, :2]
        dets_loc = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 2)  # [b, 8732,4]
        # clip bounding box
        dets_loc[:, :, 0::2] = dets_loc[:, :, 0::2].clamp(min=0, max=input_size[0] - 1).div(input_size[0])
        dets_loc[:, :, 1::2] = dets_loc[:, :, 1::2].clamp(min=0, max=input_size[1] - 1).div(input_size[1])
        dets_conf = nn.functional.softmax(pred_conf, dim=-1)
        return pred_loc, pred_conf, dets_loc, dets_conf


class LFFDv2(nn.Module):
    def __init__(self, configer):
        super(LFFDv2, self).__init__()
        self.configer = configer
        # conv block 1 ---------------------------------------------------------------------------------------
        self.conv1 = nn.Conv2d(3, num_filters_list[1], kernel_size=3, stride=(2, 2), padding=(0, 0))
        self.relu1 = nn.ReLU()
        # conv block 2 ----------------------------------------------------------------------------------------
        self.conv2 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(2, 2), padding=(0, 0))
        self.relu2 = nn.ReLU()
        # conv block 3 ----------------------------------------------------------------------------------------
        self.conv3 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        # conv block 4 ----------------------------------------------------------------------------------------
        self.conv4 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()

        # conv block 5 ----------------------------------------------------------------------------------------
        self.conv5 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu5 = nn.ReLU()
        # conv block 6 ----------------------------------------------------------------------------------------
        self.conv6 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu6 = nn.ReLU()

        # conv block 7 ----------------------------------------------------------------------------------------
        self.conv7 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu7 = nn.ReLU()
        # conv block 8 ----------------------------------------------------------------------------------------
        self.conv8 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu8 = nn.ReLU()

        # conv block 9 ----------------------------------------------------------------------------------------
        self.conv9 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(2, 2), padding=(0, 0))
        self.relu9 = nn.ReLU()
        # conv block 10 ----------------------------------------------------------------------------------------
        self.conv10 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu10 = nn.ReLU()
        # conv block 11 ----------------------------------------------------------------------------------------
        self.conv11 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu11 = nn.ReLU()

        # conv block 12 ----------------------------------------------------------------------------------------
        self.conv12 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(2, 2), padding=(0, 0))
        self.relu12 = nn.ReLU()
        # conv block 13 ----------------------------------------------------------------------------------------
        self.conv13 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu13 = nn.ReLU()
        # conv block 14 ----------------------------------------------------------------------------------------
        self.conv14 = nn.Conv2d(num_filters_list[1], num_filters_list[1], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu14 = nn.ReLU()

        # conv block 15 ----------------------------------------------------------------------------------------
        self.conv15 = nn.Conv2d(num_filters_list[1], num_filters_list[2], kernel_size=3, stride=(2, 2), padding=(0, 0))
        self.relu15 = nn.ReLU()
        # conv block 16 ----------------------------------------------------------------------------------------
        self.conv16 = nn.Conv2d(num_filters_list[2], num_filters_list[2], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu16 = nn.ReLU()
        # conv block 17 ----------------------------------------------------------------------------------------
        self.conv17 = nn.Conv2d(num_filters_list[2], num_filters_list[2], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu17 = nn.ReLU()
        # conv block 18 ----------------------------num_nonzero--------------------------------------------------------
        self.conv18 = nn.Conv2d(num_filters_list[2], num_filters_list[2], kernel_size=3, stride=(2, 2), padding=(0, 0))
        self.relu18 = nn.ReLU()
        # conv block 19 ----------------------------------------------------------------------------------------
        self.conv19 = nn.Conv2d(num_filters_list[2], num_filters_list[2], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu19 = nn.ReLU()
        # conv block 20 ----------------------------------------------------------------------------------------
        self.conv20 = nn.Conv2d(num_filters_list[2], num_filters_list[2], kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu20 = nn.ReLU()

        self.ssd_detection_layer = DetectionLayer(configer)
        self.ssd_target_generator = SSDTargetGenerator(configer)
        self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))

    def forward(self, data_dict):
        c1 = self.conv1(data_dict['img'])
        r1 = self.relu1(c1)
        c2 = self.conv2(r1)
        r2 = self.relu2(c2)
        c3 = self.conv3(r2)
        r3 = self.relu3(c3)
        c4 = self.conv4(r3)
        c4 = c2 + c4
        r4 = self.relu4(c4)

        c5 = self.conv5(r4)
        r5 = self.relu5(c5)
        c6 = self.conv6(r5)
        c6 = c4 + c6
        r6 = self.relu6(c6)

        c7 = self.conv7(r6)
        r7 = self.relu7(c7)
        c8 = self.conv8(r7)
        c8 = c6 + c8
        r8 = self.relu8(c8)

        c9 = self.conv9(r8)
        r9 = self.relu9(c9)
        c10 = self.conv10(r9)
        r10 = self.relu10(c10)
        c11 = self.conv11(r10)
        c11 = c9 + c11
        r11 = self.relu11(c11)

        c12 = self.conv12(c11)
        r12 = self.relu12(c12)
        c13 = self.conv13(r12)
        r13 = self.relu13(c13)
        c14 = self.conv14(r13)
        c14 = c12 + c14
        r14 = self.relu14(c14)

        c15 = self.conv15(r14)
        r15 = self.relu15(c15)
        c16 = self.conv16(r15)
        r16 = self.relu16(c16)
        c17 = self.conv17(r16)
        c17 = c15 + c17
        r17 = self.relu17(c17)

        c18 = self.conv18(r17)
        r18 = self.relu18(c18)
        c19 = self.conv19(r18)
        r19 = self.relu19(c19)
        c20 = self.conv20(r19)
        c20 = c18 + c20
        r20 = self.relu20(c20)

        out = [r14]
        pred_loc, pred_conf, dets_loc, dets_conf = self.ssd_detection_layer(out, data_dict)
        out_dict = dict(loc=dets_loc, conf=dets_conf)
        if self.configer.get('phase') == 'test':
            return out_dict

        loc_targets, conf_targets = self.ssd_target_generator(out, data_dict)
        loss_dict = dict()
        if 'multibox_loss' in self.valid_loss_dict:
            loss_dict['multibox_loss'] = dict(
                params=[pred_loc, pred_conf, loc_targets, conf_targets],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['multibox_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['multibox_loss']])
            )
        return out_dict, loss_dict
