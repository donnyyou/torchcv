#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


import os
import torch
import torch.nn as nn

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

import lib.model.base as base
from lib.tools.util.logger import Logger as Log


class ModuleHelper(object):

    @staticmethod
    def get_backbone(backbone, pretrained=None, **kwargs):
        model = base.__dict__[backbone](**kwargs)
        model = ModuleHelper.load_model(model, pretrained=pretrained, all_match=False)
        return model
        
    @staticmethod
    def BNReLU(num_features, norm_type=None, **kwargs):
        if norm_type == 'batchnorm':
            return nn.Sequential(
                nn.BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif norm_type == 'encsync_batchnorm':
            from encoding.nn import SyncBatchNorm
            return nn.Sequential(
                SyncBatchNorm(num_features, **kwargs),
                nn.ReLU()
            )
        elif norm_type == 'instancenorm':
            return nn.Sequential(
                nn.InstanceNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        else:
            Log.error('Not support BN type: {}.'.format(norm_type))
            exit(1)

    @staticmethod
    def BatchNorm3d(norm_type=None, ret_cls=False):
        if norm_type == 'batchnorm':
            return nn.BatchNorm3d

        elif norm_type == 'encsync_batchnorm':
            from encoding.nn import BatchNorm3d
            return BatchNorm3d

        elif norm_type == 'instancenorm':
            return nn.InstanceNorm3d
        # elif bn_type == 'inplace_abn':
        #    from extensions.ops.inplace_abn.bn import InPlaceABNSync
        #    if ret_cls:
        #        return InPlaceABNSync

        #    return functools.partial(InPlaceABNSync, activation='none')

        else:
            Log.error('Not support BN type: {}.'.format(norm_type))
            exit(1)

    @staticmethod
    def BatchNorm2d(norm_type=None, ret_cls=False):
        if norm_type == 'batchnorm':
            return nn.BatchNorm2d

        elif norm_type == 'encsync_batchnorm':
            from encoding.nn import BatchNorm2d
            return BatchNorm2d

        elif norm_type == 'instancenorm':
            return nn.InstanceNorm2d
        # elif bn_type == 'inplace_abn':
        #    from extensions.ops.inplace_abn.bn import InPlaceABNSync
        #    if ret_cls:
        #        return InPlaceABNSync

        #    return functools.partial(InPlaceABNSync, activation='none')

        else:
            Log.error('Not support BN type: {}.'.format(norm_type))
            exit(1)

    @staticmethod
    def BatchNorm1d(norm_type=None, ret_cls=False):
        if norm_type == 'batchnorm':
            return nn.BatchNorm1d

        elif norm_type == 'encsync_batchnorm':
            from encoding.nn import BatchNorm1d
            return BatchNorm1d

        elif norm_type == 'instancenorm':
            return nn.InstanceNorm1d
        # elif bn_type == 'inplace_abn':
        #    from extensions.ops.inplace_abn.bn import InPlaceABNSync
        #    if ret_cls:
        #        return InPlaceABNSync

        #    return functools.partial(InPlaceABNSync, activation='none')

        else:
            Log.error('Not support BN type: {}.'.format(norm_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, map_location='cpu'):
        if pretrained is None:
            return model

        if not os.path.exists(pretrained):
            Log.warn('{} not exists.'.format(pretrained))
            return model

        Log.info('Loading pretrained model:{}'.format(pretrained))
        if all_match:
            pretrained_dict = torch.load(pretrained, map_location=map_location)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'prefix.{}'.format(k) in model_dict:
                    load_dict['prefix.{}'.format(k)] = v
                else:
                    load_dict[k] = v

            model.load_state_dict(load_dict)

        else:
            pretrained_dict = torch.load(pretrained, map_location=map_location)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'prefix.{}'.format(k) in model_dict:
                    load_dict['prefix.{}'.format(k)] = v
                elif k in model_dict:
                    load_dict[k] = v

            Log.info('Matched Keys: {}'.format(load_dict.keys()))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join('~', '.TorchCV', 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            Log.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        Log.info('Loading pretrained model:{}'.format(cached_file))
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     bias=0,
                     distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

