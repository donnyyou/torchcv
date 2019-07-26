#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# CPN fot Parts Detection.


import torch.nn as nn
import torch
import math

from models.backbones.backbone_selector import BackboneSelector


class CPN(nn.Module):
    def __init__(self, configer):
        super(CPN, self).__init__()

        self.configer = configer
        self.backbone = BackboneSelector(configer).get_backbone()

        input_size = self.configer.get('data', 'input_size')
        stride = self.configer.get('network', 'stride')

        output_shape = (input_size[0] // stride, input_size[1] // stride)
        self.global_net = globalNet([2048, 1024, 512, 256],
                                    output_shape,
                                    self.configer.get('network', 'heatmap_out'))

        self.refine_net = refineNet(256, output_shape, self.configer.get('network', 'heatmap_out'))

    def forward(self, x):
        res_out = self.resnet101(x)

        global_re, global_out = self.global_net(res_out)

        refine_out = self.refine_net(global_re)

        return global_out, refine_out


class globalNet(nn.Module):

    def __init__(self, input_sizes, output_shape, num_points):
        super(globalNet, self).__init__()

        self.layer1_1 = self._make_layer1(input_sizes[0])
        self.layer1_2 = self._make_layer2()
        self.layer1_3 = self._make_layer3(output_shape, num_points)

        self.layer2_1 = self._make_layer1(input_sizes[1])
        self.layer2_2 = self._make_layer2()
        self.layer2_3 = self._make_layer3(output_shape, num_points)

        self.layer3_1 = self._make_layer1(input_sizes[2])
        self.layer3_2 = self._make_layer2()
        self.layer3_3 = self._make_layer3(output_shape, num_points)

        self.layer4_1 = self._make_layer1(input_sizes[3])
        self.layer4_3 = self._make_layer3(output_shape, num_points)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer1(self, input_size):

        layers = []

        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_layer2(self):

        layers = []

        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=True))

        return nn.Sequential(*layers)

    def _make_layer3(self, output_shape, num_points):

        layers = []

        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_points,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_points))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear'))

        return nn.Sequential(*layers)

    def forward(self, x):

        x1_1 = self.layer1_1(x[0])
        x1_2 = self.layer1_2(x1_1)
        x1_3 = self.layer1_3(x1_1)

        x2_1 = self.layer2_1(x[1]) + x1_2
        x2_2 = self.layer2_2(x2_1)
        x2_3 = self.layer2_3(x2_1)

        x3_1 = self.layer3_1(x[2]) + x2_2
        x3_2 = self.layer3_2(x3_1)
        x3_3 = self.layer3_3(x3_1)

        x4_1 = self.layer4_1(x[3]) + x3_2
        x4_3 = self.layer4_3(x4_1)

        return [x4_1, x3_1, x2_1, x1_1], [x4_3, x3_3, x2_3, x1_3]


class refineNet(nn.Module):

    def __init__(self, input_size, out_shape, num_points):
        super(refineNet, self).__init__()
        self.layer1 = self._make_layer1(input_size, 0, out_shape)
        self.layer2 = self._make_layer1(input_size, 1, out_shape)
        self.layer3 = self._make_layer1(input_size, 2, out_shape)
        self.layer4 = self._make_layer1(input_size, 3, out_shape)

        self.final_branch = self._make_layer2(1024, num_points)

    def _make_layer1(self, input_size, num, output_shape):

        layers = []

        for i in range(num):
            layers.append(Bottleneck(input_size, 128))

        layers.append(nn.Upsample(size=output_shape, mode='bilinear'))

        return nn.Sequential(*layers)

    def _make_layer2(self, input_size, num_points):

        layers = []

        layers.append(Bottleneck(input_size, 128))
        layers.append(nn.Conv2d(256, num_points,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_points))

        return nn.Sequential(*layers)

    def forward(self, x):

        x1 = self.layer1(x[0])
        x2 = self.layer2(x[1])
        x3 = self.layer3(x[2])
        x4 = self.layer4(x[3])

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.final_branch(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 2),
        )

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
