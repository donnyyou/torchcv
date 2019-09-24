#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# ResNet in PyTorch.


import torch
import torch.nn as nn

from model.cls.nets.cls_model import ClsModel
from model.cls.loss.loss import BASE_LOSS_DICT


class DistillModel(nn.Module):
    def __init__(self, configer):
        super(DistillModel, self).__init__()
        self.configer = configer
        self.valid_loss_dict = self.configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))
        self.main = ClsModel(self.configer, loss_dict=self.valid_loss_dict['main'], flag="main")
        self.peer = ClsModel(self.configer, loss_dict=self.valid_loss_dict['peer'], flag="peer")

        self.temperature = 1.0
        if configer.get('network.distill_method', default=None) == 'teacher-student':
            self.temperature = 10.0
            for m in self.main.parameters():
                m.requires_grad = False
        else:
            assert(configer.get('network.distill_method', default=None) in (None, 'teacher-student', 'student-student'))

    def forward(self, data_dict):
        main_out_dict, main_label_dict, main_loss_dict = self.main(data_dict)
        peer_out_dict, peer_label_dict, peer_loss_dict = self.peer(data_dict)
        out_dict = {**main_out_dict, **peer_out_dict}
        label_dict = {**main_label_dict, **peer_label_dict}
        loss_dict = {**main_loss_dict, **peer_loss_dict}
        if 'img' in data_dict:
            for i in range(len(self.configer.get('data', 'num_classes_list'))):
                if 'main_kl_loss{}'.format(i) in self.valid_loss_dict:
                    loss_dict['main_kl_loss{}'.format(i)] = dict(
                        params=[out_dict['main_out{}'.format(i)], out_dict['peer_out{}'.format(i)].detach()],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['kl_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['main_kl_loss{}'.format(i)]])
                    )
                if 'peer_kl_loss{}'.format(i) in self.valid_loss_dict:
                    loss_dict['peer_kl_loss{}'.format(i)] = dict(
                        params=[out_dict['peer_out{}'.format(i)].div(self.temperature),
                                out_dict['main_out{}'.format(i)].div(self.temperature).detach()],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['kl_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['peer_kl_loss{}'.format(i)]])
                    )

        return out_dict, label_dict, loss_dict
