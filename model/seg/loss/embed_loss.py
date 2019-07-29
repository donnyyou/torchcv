#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


import torch
import torch.nn as nn


class EmbedLoss(nn.Module):
    def __init__(self, configer):
        super(EmbedLoss, self).__init__()
        self.num_classes = configer.get('data', 'num_classes')
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, inputs, targets):
        inputs = inputs.transpose(0, 1)
        center_array = torch.zeros((self.num_classes, inputs.size()[0]), requires_grad=True).cuda()
        sim_loss = torch.Tensor([0.0]).cuda()
        sim_loss.requires_grad = True

        mask_list = list()
        for i in range(self.num_classes):
            mask = self.get_mask(targets, i).unsqueeze(0)
            sum_pixel = max(mask.sum(), 1)
            # print sum_pixel
            mask = mask.contiguous().repeat(inputs.size()[0], 1, 1, 1).byte().cuda()
            sim_input = inputs[mask]
            if sim_input.numel() == 0:
                mask_list.append(i)
                continue

            sim_input = sim_input.contiguous().view(inputs.size()[0], -1)
            center = torch.sum(sim_input, 1, keepdim=False)
            center = center / sum_pixel
            center_array[i, :] = center

            sim_input = sim_input.permute(1, 0)

            sim_label = torch.ones(sim_input.size()[0], ).float().cuda()
            sim_center = center.contiguous().view(1, -1).repeat(sim_input.size()[0], 1)
            sim_loss = sim_loss + self.cosine_loss(sim_center, sim_input, sim_label)

        diff_loss = torch.Tensor([0.0]).cuda()
        diff_loss.requires_grad = True
        for i in range(self.num_classes):
            if i in mask_list:
                continue

            label = torch.zeros(self.num_classes, ).float().cuda()
            center_dual = torch.zeros((self.num_classes, inputs.size()[0]), requires_grad=True).cuda()
            for k in range(self.num_classes):
                center_dual[k] = center_array[i]

            for j in range(self.num_classes):
                if j == i:
                    label[j] = 1
                else:
                    label[j] = -1

            diff_loss = diff_loss + self.cosine_loss(center_array, center_dual, label)

        embedding_loss = diff_loss + sim_loss
        # print embedding_loss.requires_grad
        return embedding_loss

    def get_mask(self, targets, i):
        targets_cp = torch.cuda.FloatTensor(targets.size())
        targets_cp.copy_(targets.data)
        if i == 0:
            targets_cp[targets_cp != 0] = 2
            targets_cp[targets_cp == 0] = 1
            targets_cp[targets_cp == 2] = 0
        else:
            targets_cp[targets_cp != i] = 0
            targets_cp[targets_cp == i] = 1

        return targets_cp
