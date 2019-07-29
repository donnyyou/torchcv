#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn


class OpenPose(nn.Module):
    def __init__(self, model_dict):
        super(OpenPose, self).__init__()

        self.model0 = model_dict['block_0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        paf_out = [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
        heatmap_out = [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]
        return paf_out, heatmap_out


def make_layers(layer_dict):
    layers = []

    for i in range(len(layer_dict) - 1):
        layer = layer_dict[i]
        for k in layer:
            v = layer[k]
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    layer = list(layer_dict[-1].keys())
    k = layer[0]
    v = layer_dict[-1][k]

    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]

    return nn.Sequential(*layers)


def get_open_pose_org(configer):
    paf_out = configer.get('network', 'paf_out')
    heatmap_out = configer.get('network', 'heatmap_out')

    blocks = {}

    block_0 = [{'conv1_1': [3, 64, 3, 1, 1]}, {'conv1_2': [64, 64, 3, 1, 1]}, {'pool1_stage1': [2, 2, 0]},
               {'conv2_1': [64, 128, 3, 1, 1]}, {'conv2_2': [128, 128, 3, 1, 1]}, {'pool2_stage1': [2, 2, 0]},
               {'conv3_1': [128, 256, 3, 1, 1]}, {'conv3_2': [256, 256, 3, 1, 1]}, {'conv3_3': [256, 256, 3, 1, 1]},
               {'conv3_4': [256, 256, 3, 1, 1]}, {'pool3_stage1': [2, 2, 0]}, {'conv4_1': [256, 512, 3, 1, 1]},
               {'conv4_2': [512, 512, 3, 1, 1]}, {'conv4_3_CPM': [512, 256, 3, 1, 1]},
               {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, paf_out, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]}, {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, heatmap_out, 1, 1, 0]}]

    for i in range(2, 7):
        blocks['block%d_1' % i] = [{'Mconv1_stage%d_L1' % i: [128+paf_out+heatmap_out, 128, 7, 1, 3]},
                                   {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                                   {'Mconv7_stage%d_L1' % i: [128, paf_out, 1, 1, 0]}]
        blocks['block%d_2' % i] = [{'Mconv1_stage%d_L2' % i: [128+paf_out+heatmap_out, 128, 7, 1, 3]},
                                   {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                                   {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                                   {'Mconv7_stage%d_L2' % i: [128, heatmap_out, 1, 1, 0]}]

    layers = []
    for block in block_0:
        for key in block:
            v = block[key]
            if 'pool' in key:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]

    models = {
        'block_0': nn.Sequential(*layers)
    }

    for k in blocks:
        v = blocks[k]
        models[k] = make_layers(v)

    return OpenPose(models)
