#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch
import torch.nn as nn
import torch.nn.functional as F


class LiftedStructureLoss(nn.Module):
    def __init__(self, configer):
        super(LiftedStructureLoss, self).__init__()
        self.params_dict = dict()
        if 'lifted_structure_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['lifted_structure_loss']

    @staticmethod
    def pdist(A, squared=False):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=1e-12)
        return res if squared else res.sqrt()

    def forward(self, embeddings, labels):
        d = self.pdist(embeddings, squared=False)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
        neg_i = torch.mul((self.params_dict['margin'] - d).exp(), 1 - pos).sum(1).expand_as(d).clamp(min=1e-12)
        return torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d) + 1e-8)
