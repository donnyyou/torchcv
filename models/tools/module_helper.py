#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (donnyyou@163.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch.nn as nn

from utils.tools.logger import Logger as Log


class ModuleHelper(object):

    @staticmethod
    def BatchNorm2d(bn_type='torch'):
        if bn_type == 'torchbn':
            return nn.BatchNorm2d

        elif bn_type == 'syncbn':
            from extensions.layers.syncbn.module import BatchNorm2d
            return BatchNorm2d

        else:
            Log.error('Not support BN type: {}.'.format(bn_type))
            exit(1)


