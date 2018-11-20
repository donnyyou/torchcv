#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from utils.tools.logger import Logger as Log


model_urls = {
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

CONFIG_DICT = {
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg13_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512],
    'vgg16_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512],
    'vgg19_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512],
    'vgg13_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'vgg16_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512],
    'vgg19_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, cfg_name, vgg_cfg=None, bn=False):
        super(VGG, self).__init__()
        self.num_features = 512
        vgg_cfg = vgg_cfg if vgg_cfg is not None else CONFIG_DICT[cfg_name]
        self.features = make_layers(vgg_cfg, bn)

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        x = self.features(x)
        return x


class VGGModels(object):

    def __init__(self, configer):
        self.configer = configer

    def vgg(self, vgg_cfg=None):
        """Constructs a ResNet-18 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        backbone = self.configer.get('network', 'backbone')
        model = VGG(cfg_name=backbone, vgg_cfg=vgg_cfg, bn=False)
        if self.configer.get('network', 'pretrained') or self.configer.get('network', 'pretrained_model') is not None:
            if self.configer.get('network', 'pretrained_model') is not None:
                Log.info('Loading pretrained model:{}'.format(self.configer.get('network', 'pretrained_model')))
                pretrained_dict = torch.load(self.configer.get('network', 'pretrained_model'))
            else:
                pretrained_dict = self.load_url(model_urls[backbone.split('_')[0]])

            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                new_key = k
                if 'features' not in k:
                    new_key = 'features.{}'.format(k)

                if new_key in model_dict:
                    load_dict[new_key] = v

            Log.info('Matched Keys: {}'.format(load_dict.keys()))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    def vgg_bn(self, vgg_cfg=None):
        backbone = self.configer.get('network', 'backbone')
        model = VGG(cfg_name=backbone, vgg_cfg=vgg_cfg, bn=True)
        if self.configer.get('network', 'pretrained') or self.configer.get('network', 'pretrained_model') is not None:
            if self.configer.get('network', 'pretrained_model') is not None:
                Log.info('Loading pretrained model:{}'.format(self.configer.get('network', 'pretrained_model')))
                pretrained_dict = torch.load(self.configer.get('network', 'pretrained_model'))
            else:
                pretrained_dict = self.load_url(model_urls['{}_bn'.format(backbone.split('_')[0])])

            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                new_key = k
                if 'features' not in k:
                    new_key = 'features.{}'.format(k)

                if new_key in model_dict:
                    load_dict[new_key] = v

            Log.info('Matched Keys: {}'.format(load_dict.keys()))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    def load_url(self, url, map_location=None):
        model_dir = os.path.join(self.configer.get('project_dir'), 'models/backbones/vgg/pretrained')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            Log.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        Log.info('Loading pretrained model:{}'.format(cached_file))
        return torch.load(cached_file, map_location=map_location)


if __name__ == "__main__":
    pass
