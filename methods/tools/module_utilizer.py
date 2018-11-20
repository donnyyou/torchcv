#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# Some methods used by main methods.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import gather as torch_gather

from utils.tools.logger import Logger as Log


class ModuleUtilizer(object):

    def __init__(self, configer):
        self.configer = configer
        self._init()

    def _init(self):
        self.configer.add_key_value(['iters'], 0)
        self.configer.add_key_value(['last_iters'], 0)
        self.configer.add_key_value(['epoch'], 0)
        self.configer.add_key_value(['last_epoch'], 0)
        self.configer.add_key_value(['max_performance'], 0.0)
        self.configer.add_key_value(['performance'], 0.0)
        self.configer.add_key_value(['min_val_loss'], 9999.0)
        self.configer.add_key_value(['val_loss'], 9999.0)
        self.configer.add_key_value(['network', 'parallel'], False)
        if self.configer.is_empty('network', 'bn_type'):
            self.configer.add_key_value(['network', 'bn_type'], 'torchbn')

        if len(self.configer.get('gpu')) == 1:
            self.configer.update_value(['network', 'bn_type'], 'torchbn')

        Log.info('BN Type is {}.'.format(self.configer.get('network', 'bn_type')))

    def to_device(self, *params):
        device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        return_list = list()
        for i in range(len(params)):
            return_list.append(params[i].to(device))

        return return_list[0] if len(params) == 1 else return_list

    def _make_parallel(self, net):
        if self.configer.get('network', 'bn_type') == 'syncbn':
            assert len(self.configer.get('gpu')) > 1
            from extensions.layers.syncbn.parallel import DataParallelModel
            self.configer.update_value(['network', 'parallel'], True)
            return DataParallelModel(net)

        elif len(self.configer.get('gpu')) > 1:
            self.configer.update_value(['network', 'parallel'], True)
            return nn.DataParallel(net)

        else:
            return net

    def load_net(self, net):
        if self.configer.get('gpu') is not None:
            net = self._make_parallel(net)

        net = net.to(torch.device('cpu' if self.configer.get('gpu') is None else 'cuda'))

        if self.configer.get('network', 'resume') is not None:
            resume_dict = torch.load(self.configer.get('network', 'resume'))
            if 'state_dict' in resume_dict:
                checkpoint_dict = resume_dict['state_dict']

            elif 'model' in resume_dict:
                checkpoint_dict = resume_dict['model']

            else:
                checkpoint_dict = resume_dict

            net_dict = net.state_dict()

            not_match_list = list()
            for key, value in checkpoint_dict.items():
                if key.split('.')[0] == 'module':
                    module_key = key
                    norm_key = '.'.join(key.split('.')[1:])
                else:
                    module_key = 'module.{}'.format(key)
                    norm_key = key

                if self.configer.get('network', 'parallel'):
                    key = module_key
                else:
                    key = norm_key

                if net_dict[key].size() == value.size():
                    net_dict[key] = value
                else:
                    not_match_list.append(key)

            if self.configer.get('network', 'resume_level') == 'full':
                assert len(not_match_list) == 0

            elif self.configer.get('network', 'resume_level') == 'part':
                Log.info('Not Matched Keys: {}'.format(not_match_list))

            else:
                Log.error('Resume Level: {} is invalid.'.format(self.configer.get('network', 'resume_level')))
                exit(1)

            if self.configer.get('network', 'resume_continue'):
                self.configer.update_value(['epoch'], resume_dict['config_dict']['epoch'])
                self.configer.update_value(['iters'], resume_dict['config_dict']['iters'])
                self.configer.update_value(['performance'], resume_dict['config_dict']['performance'])
                self.configer.update_value(['val_loss'], resume_dict['config_dict']['val_loss'])

            net.load_state_dict(net_dict)

        return net

    def save_net(self, net, save_mode='iters'):
        state = {
            'config_dict': self.configer.to_dict(),
            'state_dict': net.state_dict(),
        }
        if self.configer.get('checkpoints', 'checkpoints_root') is None:
            checkpoints_dir = os.path.join(self.configer.get('project_dir'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))
        else:
            checkpoints_dir = os.path.join(self.configer.get('checkpoints', 'checkpoints_root'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        if save_mode == 'performance':
            if self.configer.get('performance') > self.configer.get('max_performance'):
                latest_name = '{}_max_performance.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update_value(['max_performance'], self.configer.get('performance'))

        elif save_mode == 'val_loss':
            if self.configer.get('val_loss') < self.configer.get('min_val_loss'):
                latest_name = '{}_min_loss.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update_value(['min_val_loss'], self.configer.get('val_loss'))

        elif save_mode == 'iters':
            if self.configer.get('iters') - self.configer.get('last_iters') >= \
                    self.configer.get('checkpoints', 'save_iters'):
                latest_name = '{}_iters{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('iters'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update_value(['last_iters'], self.configer.get('iters'))

        elif save_mode == 'epoch':
            if self.configer.get('epoch') - self.configer.get('last_epoch') >= \
                    self.configer.get('checkpoints', 'save_epoch'):
                latest_name = '{}_epoch{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('epoch'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update_value(['last_epoch'], self.configer.get('epoch'))

        else:
            Log.error('Metric: {} is invalid.'.format(save_mode))
            exit(1)

    def freeze_bn(self, net, syncbn=False):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

            if syncbn:
                from extensions.layers.syncbn.module import BatchNorm2d, BatchNorm1d
                if isinstance(m, BatchNorm2d) or isinstance(m, BatchNorm1d):
                    m.eval()

    def clip_grad(self, model, max_grad=10.):
        """Computes a gradient clipping coefficient based on gradient norm."""
        total_norm = 0
        for p in model.parameters():
            if p.requires_grad:
                modulenorm = p.grad.data.norm()
                total_norm += modulenorm ** 2

        total_norm = math.sqrt(total_norm)

        norm = max_grad / max(total_norm, max_grad)
        for p in model.parameters():
            if p.requires_grad:
                p.grad.mul_(norm)

    def gather(self, outputs, target_device=None, dim=0):
        r"""
        Gathers tensors from different GPUs on a specified device
          (-1 means the CPU).
        """
        if self.configer.get('network', 'bn_type') == 'syncbn':
            if target_device is None:
                target_device = list(range(torch.cuda.device_count()))[0]

            return torch_gather(outputs, target_device, dim=dim)

        else:
            assert isinstance(outputs, torch.Tensor)
            return outputs



