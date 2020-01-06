#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# ResNet in PyTorch.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cls.loss.loss import BASE_LOSS_DICT
from model.cls.utils.metric_linear import LINEAR_DICT
from lib.model.module_helper import ModuleHelper


class ClsModel(nn.Module):
    def __init__(self, configer, loss_dict=None, flag=""):
        super(ClsModel, self).__init__()
        self.configer = configer
        self.flag = flag if len(flag) == 0 else "{}_".format(flag)
        self.backbone = ModuleHelper.get_backbone(
            backbone=configer.get('network.{}backbone'.format(self.flag)),
            pretrained=configer.get('network.{}pretrained'.format(self.flag))
        )

        self.reduction = None
        fc_dim_out = configer.get('network.{}fc_dim'.format(self.flag), default=None)
        fc_dim = self.backbone.num_features
        if fc_dim_out is not None:
            self.reduction = nn.Conv2d(self.backbone.num_features, fc_dim_out, 1)
            fc_dim = fc_dim_out

        self.linear_list = nn.ModuleList()
        linear_type = configer.get('network', '{}linear_type'.format(self.flag), default='linear')
        self.fc = LINEAR_DICT[linear_type](fc_dim, configer.get('data.num_classes'))

        self.embed = None
        if configer.get('network.{}embed'.format(self.flag), default=False):
            feat_dim = configer.get('network', '{}feat_dim'.format(self.flag))
            self.embed = nn.Sequential(
                nn.Linear(fc_dim, feat_dim),
                nn.BatchNorm1d(feat_dim)
            )

        self.bn = None
        if configer.get('network.{}bn'.format(self.flag), default=False):
            self.bn = nn.BatchNorm1d(fc_dim)
            nn.init.zeros_(self.bn.bias)
            self.bn.bias.requires_grad = False

        self.valid_loss_dict = configer.get('loss.loss_weights',
                                            configer.get('loss.loss_type')) if loss_dict is None else loss_dict

    @staticmethod
    def mixup_data(data_dict, alpha=0.2):
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.

        batch_size = data_dict['img'].size(0)
        index = torch.randperm(batch_size).to(data_dict['img'].device)
        data_dict['img'] = lam * data_dict['img'] + (1 - lam) * data_dict[index, ...]
        data_dict['label_a'], data_dict['label_b'] = data_dict['label'], data_dict['label'][index]
        return data_dict, lam

    def forward(self, data_dict):
        out_dict = dict()
        label_dict = dict()
        loss_dict = dict()
        lam = 1.0
        if self.configer.get('data.mixup', default=False):
            data_dict, lam = self.mixup_data(data_dict, self.configer.get('data.mixup_alpha'))

        x = self.backbone(data_dict['img'])
        x = x[-1] if isinstance(x, (list, tuple)) else x
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.reduction(x) if self.reduction else x
        x = x.view(x.size(0), -1)
        fc = self.bn(x) if self.bn else x
        out = self.fc(fc, data_dict['label'])
        out_dict['{}out'.format(self.flag)] = out
        label_dict['{}out'.format(self.flag)] = data_dict['label']
        if 'ce_loss' in self.valid_loss_dict:
            loss_dict['{}ce_loss'.format(self.flag)] = dict(
                params=[out, data_dict['label']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_loss']])
            )
        if 'soft_ce_loss' in self.valid_loss_dict:
            loss_dict['{}soft_ce_loss'.format(self.flag)] = dict(
                params=[out, data_dict['label'], self.configer.get('data.num_classes')],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['soft_ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['soft_ce_loss']])
            )
        if 'mixup_ce_loss' in self.valid_loss_dict:
            assert 'label_a' in data_dict and 'label_b' in data_dict
            loss_dict['{}mixup_ce_loss'.format(self.flag)] = dict(
                params=[out, data_dict['label_a'], data_dict['label_b'], lam],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['mixup_ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['mixup_ce_loss']])
            )
        if 'mixup_soft_ce_loss' in self.valid_loss_dict:
            assert 'label_a' in data_dict and 'label_b' in data_dict
            loss_dict['{}mixup_soft_ce_loss'.format(self.flag)] = dict(
                params=[out, data_dict['label_a'], data_dict['label_b'], self.configer.get('data.num_classes'), lam],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['mixup_soft_ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['mixup_soft_ce_loss']])
            )

        feat = self.embed(x) if self.embed else x
        for i in range(len(self.linear_list)):
            if 'tri_loss' in self.valid_loss_dict:
                loss_dict['{}tri_loss'.format(self.flag)] = dict(
                    params=[feat, data_dict['label']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['hard_triplet_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['tri_loss']])
                )
            if 'ls_loss' in self.valid_loss_dict:
                loss_dict['{}ls_loss'.format(self.flag)] = dict(
                    params=[feat, data_dict['label']],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['lifted_structure_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['ls_loss']])
                )

        return out_dict, label_dict, loss_dict
