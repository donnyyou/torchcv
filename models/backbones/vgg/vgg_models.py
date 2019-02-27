#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch.nn as nn

from models.tools.module_helper import ModuleHelper


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
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'), all_match=False)
        return model

    def vgg_bn(self, vgg_cfg=None):
        backbone = self.configer.get('network', 'backbone')
        model = VGG(cfg_name=backbone, vgg_cfg=vgg_cfg, bn=True)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'), all_match=False)
        return model


if __name__ == "__main__":
    pass
