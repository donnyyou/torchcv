#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# VGG11/13/16/19 in Pytorch.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import torch
import torch.nn as nn


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG19(nn.Module):
    def __init__(self, configer):
        super(VGG19, self).__init__()
        self.configer = configer
        self.features = make_layers(cfg['vgg19'])
        width = self.configer.get('data', 'input_size')[0] // self.configer.get('network', 'stride')
        height = self.configer.get('data', 'input_size')[1] // self.configer.get('network', 'stride')
        self.avg_pool = nn.AvgPool2d(kernel_size=(height, width))
        self.classifier = nn.Linear(512, self.configer.get('data', 'num_classes'))

        if not self.configer.is_empty('phase') and \
                        self.configer.get('phase') == 'train' and \
                        self.configer.get('network', 'pretrained') is not None:
            self._load_pretrained_weight(torch.load(self.configer.get('network', 'pretrained')))

    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _load_pretrained_weight(self, net):
        new_state_dict = OrderedDict()
        model_dict = self.state_dict()
        for k, v in net.items():
            k_name = k.split('.')
            if k_name[0] == 'features' and v.size() == model_dict[k].size():
                new_state_dict[k] = v

        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)


if "__name__" == "__main__":
    pass
