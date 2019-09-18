#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# ResNet in PyTorch.


import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tools.module_helper import ModuleHelper
from model.backbones.backbone_selector import BackboneSelector
from model.cls.loss.loss import BASE_LOSS_DICT
from model.cls.utils.metric_linear import LINEAR_DICT


LOSS_TYPE = {
    'ce_loss': {
        'ce_loss0': 1.0, 'ce_loss1': 0.01
    },
    'trice_loss': {
        'ce_loss0': 1.0, 'ce_loss1': 0.01, 'tri_loss0': 0.1, 'tri_loss1': 0.01
    },
    'lsce_loss': {
        'ce_loss0': 1.0, 'ce_loss1': 0.01, 'ls_loss0': 0.1, 'ls_loss1': 0.01
    }
}


class ClsModel(nn.Module):
    def __init__(self, configer, loss_dict=None, flag=""):
        super(ClsModel, self).__init__()
        self.configer = configer
        self.flag = flag if len(flag) == 0 else "{}_".format(flag)
        self.backbone = BackboneSelector(configer).get_backbone(
            backbone_type=configer.get('network.{}backbone'.format(self.flag)),
            pretrained_model=configer.get('network.{}pretrained'.format(self.flag)),
            rm_last_stride=configer.get('network.{}rm_last_stride'.format(self.flag), default=False)
        )

        self.reduction = None
        fc_dim_out = configer.get('network.{}fc_dim'.format(self.flag), default=None)
        fc_dim = self.backbone.num_features
        if fc_dim_out is not None:
            self.reduction = nn.Conv2d(self.backbone.num_features, fc_dim_out, 1)
            fc_dim = fc_dim_out

        self.linear_list = nn.ModuleList()
        linear_type = configer.get('network', '{}linear_type'.format(self.flag))
        for num_classes in configer.get('data.num_classes_list'):
            self.linear_list.append(LINEAR_DICT[linear_type](fc_dim, num_classes))

        self.embed = None
        if configer.get('network.{}embed'.format(self.flag), default=True):
            feat_dim = configer.get('network', '{}feat_dim'.format(self.flag))
            self.embed = nn.Sequential(
                nn.Linear(fc_dim, feat_dim),
                nn.BatchNorm1d(feat_dim)
            )

        self.bn = nn.BatchNorm1d(fc_dim)
        nn.init.zeros_(self.bn.bias)
        self.bn.bias.requires_grad = False

        self.valid_loss_dict = LOSS_TYPE[configer.get('loss', 'loss_type')] if loss_dict is None else loss_dict

    def forward(self, data_dict):
        out_dict = dict()
        label_dict = dict()
        loss_dict = dict()
        x_ = self.backbone(data_dict['img'])
        if 'img' in data_dict:
            x = x_[:data_dict['img'].size(0)]
            x = F.adaptive_avg_pool2d(x, 1)
            x = self.reduction(x) if self.reduction else x
            x = x.view(x.size(0), -1)
            fc = self.bn(x)
            for i in range(len(self.linear_list)):
                sub_out = self.linear_list[i](fc, data_dict['label'][:, i])
                out_dict['{}out{}'.format(self.flag, i)] = sub_out
                label_dict['{}out{}'.format(self.flag, i)] = data_dict['label'][:, i]
                if 'ce_loss{}'.format(i) in self.valid_loss_dict:
                    loss_dict['{}ce_loss{}'.format(self.flag, i)] = dict(
                        params=[sub_out, data_dict['label'][:, i]],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_loss{}'.format(i)]])
                    )

            feat = self.embed(x) if self.embed else x
            for i in range(len(self.linear_list)):
                if 'tri_loss{}'.format(i) in self.valid_loss_dict:
                    loss_dict['{}tri_loss{}'.format(self.flag, i)] = dict(
                        params=[feat, data_dict['label'][:, i]],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['hard_triplet_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['tri_loss{}'.format(i)]])
                    )
                if 'ls_loss{}'.format(i) in self.valid_loss_dict:
                    loss_dict['{}ls_loss{}'.format(self.flag, i)] = dict(
                        params=[feat, data_dict['label'][:, i]],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['lifted_structure_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['ls_loss{}'.format(i)]])
                    )

        return out_dict, label_dict, loss_dict
