#!/usr/bin/env python
#encoding=utf8
#########################################################################
# Author:
# Created Time: Mon Sep 23 14:22:52 2019
# File Name: dfnet_models.py
# Description:
#########################################################################


import math
import torch
import torch.nn as nn


__all__ = ['dfnetv1', 'dfnetv2']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DFNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(DFNetV1, self).__init__()
        self.inplanes = 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage2 = self._make_layer(64, 3, stride=2)
        self.stage3 = self._make_layer(128, 3, stride=2)
        self.stage4 = self._make_layer(256, 3, stride=2)
        self.stage5 = self._make_layer(512, 1, stride=1)
        self.num_features = 512 * BasicBlock.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)  # 4x32
        x = self.stage2(x)  # 8x64
        x = self.stage3(x)  # 16x128
        x = self.stage4(x)  # 32x256
        x = self.stage5(x)  # 32x512
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DFNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(DFNetV2, self).__init__()
        self.inplanes = 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage2_1 = self._make_layer(64, 2, stride=2)
        self.stage2_2 = self._make_layer(128, 1, stride=1)
        self.stage3_1 = self._make_layer(128, 10, stride=2)
        self.stage3_2 = self._make_layer(256, 1, stride=1)
        self.stage4_1 = self._make_layer(256, 4, stride=2)
        self.stage4_2 = self._make_layer(512, 2, stride=1)
        self.num_features = 512 * BasicBlock.expansion

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)  # 4x32
        x = self.stage2_1(x)  # 8x64
        x = self.stage2_2(x)  # 8x64
        x = self.stage3_1(x)  # 16x128
        x = self.stage3_2(x)  # 16x128
        x = self.stage4_1(x)  # 32x256
        x = self.stage4_2(x)  # 32x256
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def dfnetv1():
    model = DFNetV1(num_classes=1000)
    return model


def dfnetv2():
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = DFNetV2(num_classes=1000)
    return model

if __name__ == "__main__":
    model = DFNetV1(1000)
    print(model)
    a = torch.rand(1, 3, 224, 224)
    print(model(a).size())
