#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)


import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, configer):
        super(MLP, self).__init__()
        self.configer = configer
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
        )
        self.fc = nn.Linear(12, self.configer.get('data', 'num_classes'))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.fc(x)
        return x
