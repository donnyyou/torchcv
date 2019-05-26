#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


import torch
import torch.nn as nn

from models.seg.loss.base_moduls import BaseCELoss, BaseEncLoss, BaseOhemCELoss


class SegCELoss(nn.Module):
    def __init__(self, configer=None):
        super(SegCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = BaseCELoss(self.configer)

    def forward(self, out_dict, data_dict, **kwargs): 
        loss = self.ce_loss(out_dict['out'], data_dict['labelmap'])
        return dict(loss=loss)


class SegOhemCELoss(nn.Module):
    def __init__(self, configer=None):
        super(SegOhemCELoss, self).__init__()
        self.configer = configer
        self.ohem_ce_loss = BaseOhemCELoss(self.configer)

    def forward(self, out_dict, data_dict, **kwargs): 
        loss = self.ohem_ce_loss(out_dict['out'], data_dict['labelmap'])
        return dict(loss=loss)


class SegAuxOhemCELoss(nn.Module):
    def __init__(self, configer=None):
        super(SegAuxOhemCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = BaseCELoss(self.configer)
        self.ohem_ce_loss = BaseOhemCELoss(self.configer)

    def forward(self, out_dict, data_dict, **kwargs):
        seg_loss = self.ohem_ce_loss(out_dict['out'], data_dict['labelmap'])
        aux_loss = self.ce_loss(out_dict['aux_out'], data_dict['labelmap'])
        loss = self.configer.get('loss', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('loss', 'loss_weights')['aux_loss'] * aux_loss
        return dict(loss=loss, aux_loss=aux_loss, seg_loss=seg_loss)


class SegAuxCELoss(nn.Module):
    def __init__(self, configer=None):
        super(SegAuxCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = BaseCELoss(self.configer)

    def forward(self, out_dict, data_dict, **kwargs):
        seg_loss = self.ce_loss(out_dict['out'], data_dict['labelmap'])
        aux_loss = self.ce_loss(out_dict['aux_out'], data_dict['labelmap'])
        loss = self.configer.get('loss', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('loss', 'loss_weights')['aux_loss'] * aux_loss
        return dict(loss=loss, aux_loss=aux_loss, seg_loss=seg_loss)


class SegAuxEncCELoss(nn.Module):
    def __init__(self, configer=None):
        super(SegAuxEncCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = BaseCELoss(self.configer)
        self.se_loss = BaseEncLoss(self.configer)

    def forward(self, out_dict, data_dict, **kwargs):
        seg_loss = self.ce_loss(out_dict['out'], data_dict['labelmap'])
        aux_loss = self.ce_loss(out_dict['aux_out'], data_dict['labelmap'])
        enc_loss = self.enc_loss(out_dict['enc_out'], data_dict['labelmap'])
        loss = self.configer.get('loss', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('loss', 'loss_weights')['aux_loss'] * aux_loss
        loss = loss + self.configer.get('loss', 'loss_weights')['enc_loss'] * enc_loss
        return dict(loss=loss, aux_loss=aux_loss, seg_loss=seg_loss, enc_loss=enc_loss)


if __name__ == "__main__":
    inputs = torch.ones((3, 5, 6, 6)).cuda()
    targets = torch.ones((3, 6, 6)).cuda()
