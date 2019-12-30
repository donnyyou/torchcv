#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some runner used by main runner.


import math
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import gather as torch_gather

from lib.tools.helper.dist_helper import DistHelper
from lib.tools.util.logger import Logger as Log


class RunnerHelper(object):

    @staticmethod
    def to_device(runner, in_data):
        device = torch.device('cpu' if runner.configer.get('gpu') is None else 'cuda')
        if isinstance(in_data, (list, tuple)):
            return [RunnerHelper.to_device(runner, item) for item in in_data]

        if isinstance(in_data, dict):
            return {k: RunnerHelper.to_device(runner, v) for k, v in in_data.items()}

        return in_data.to(device) if isinstance(in_data, torch.Tensor) else in_data

    @staticmethod
    def _make_parallel(runner, net):
        if runner.configer.get('network.distributed', default=False):
            local_rank = runner.configer.get('local_rank')
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            if runner.configer.get('network.syncbn', default=False):
                Log.info('Converting syncbn model...')
                net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

            net = nn.parallel.DistributedDataParallel(net.cuda(), find_unused_parameters=True,
                                                      device_ids=[local_rank], output_device=local_rank)
            # if runner.configer.get('network.syncbn', default=False):
            #     Log.info('Converting syncbn model...')
            #     from apex.parallel import convert_syncbn_model
            #     net = convert_syncbn_model(net)
            # from apex.parallel import DistributedDataParallel
            # net = DistributedDataParallel(net.cuda(), delay_allreduce=True)
            return net

        net = net.to(torch.device('cpu' if runner.configer.get('gpu') is None else 'cuda'))
        from lib.parallel.data_parallel import ParallelModel
        return ParallelModel(net, gather_=runner.configer.get('network', 'gather'))

    @staticmethod
    def load_net(runner, net, model_path=None, map_location='cpu'):
        if model_path is not None or runner.configer.get('network', 'resume') is not None:
            resume_path = runner.configer.get('network', 'resume')
            resume_path = model_path if model_path is not None else resume_path
            Log.info('Resuming from {}'.format(resume_path))
            resume_dict = torch.load(resume_path, map_location=map_location)
            if 'state_dict' in resume_dict:
                checkpoint_dict = resume_dict['state_dict']

            elif 'model' in resume_dict:
                checkpoint_dict = resume_dict['model']

            elif isinstance(resume_dict, OrderedDict):
                checkpoint_dict = resume_dict

            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(runner.configer.get('network', 'resume')))

            # load state_dict
            if hasattr(net, 'module'):
                RunnerHelper.load_state_dict(net.module, checkpoint_dict,
                                             runner.configer.get('network', 'resume_strict'))
            else:
                RunnerHelper.load_state_dict(net, checkpoint_dict, runner.configer.get('network', 'resume_strict'))

            if runner.configer.get('network', 'resume_continue'):
                # runner.configer.resume(resume_dict['config_dict'])
                runner.runner_state = resume_dict['runner_state']

        net = RunnerHelper._make_parallel(runner, net)
        return net

    @staticmethod
    def load_state_dict(module, state_dict, strict=False):
        """Load state_dict to a module.
        This method is modified from :meth:`torch.nn.Module.load_state_dict`.
        Default value for ``strict`` is set to ``False`` and the message for
        param mismatch will be shown even if strict is False.
        Args:
            module (Module): Module that receives the state_dict.
            state_dict (OrderedDict): Weights.
            strict (bool): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        """

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        unexpected_keys = []
        unmatched_keys = []
        own_state = module.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception:
                if strict:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(),
                                               param.size()))
                else:
                    unmatched_keys.append(name)

        missing_keys = set(own_state.keys()) - set(state_dict.keys())

        err_msg = []
        if unexpected_keys:
            err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
        if missing_keys:
            err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
        if unexpected_keys:
            err_msg.append('unmatched keys in source state_dict: {}\n'.format(', '.join(unmatched_keys)))
        err_msg = '\n'.join(err_msg)
        if err_msg:
            if strict:
                raise RuntimeError(err_msg)
            else:
                Log.warn(err_msg)

    @staticmethod
    def save_net(runner, net, performance=None, val_loss=None, iters=None, epoch=None):
        state = {
            'config_dict': runner.configer.to_dict(),
            'state_dict': net.state_dict(),
            'runner_state': runner.runner_state
        }
        if runner.configer.get('network', 'checkpoints_root') is None:
            checkpoints_dir = os.path.join(runner.configer.get('project_dir'),
                                           runner.configer.get('network', 'checkpoints_dir'))
        else:
            checkpoints_dir = os.path.join(runner.configer.get('network', 'checkpoints_root'),
                                           runner.configer.get('network', 'checkpoints_dir'))

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        latest_name = '{}_latest.pth'.format(runner.configer.get('network', 'checkpoints_name'))
        torch.save(state, os.path.join(checkpoints_dir, latest_name))
        if performance is not None:
            if performance > runner.runner_state['max_performance']:
                latest_name = '{}_max_performance.pth'.format(runner.configer.get('network', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                runner.runner_state['max_performance'] = performance

        if val_loss is not None:
            if val_loss < runner.runner_state['min_val_loss']:
                latest_name = '{}_min_loss.pth'.format(runner.configer.get('network', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                runner.runner_state['min_val_loss'] = val_loss

        if iters is not None:
            latest_name = '{}_iters{}.pth'.format(runner.configer.get('network', 'checkpoints_name'), iters)
            torch.save(state, os.path.join(checkpoints_dir, latest_name))

        if epoch is not None:
            latest_name = '{}_epoch{}.pth'.format(runner.configer.get('network', 'checkpoints_name'), epoch)
            torch.save(state, os.path.join(checkpoints_dir, latest_name))

    @staticmethod
    def freeze_bn(net, norm_type=None):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
                m.eval()

            if norm_type is not None:
                from model.tools.module_helper import ModuleHelper
                if isinstance(m, ModuleHelper.BatchNorm2d(norm_type=norm_type, ret_cls=True)) \
                        or isinstance(m, ModuleHelper.BatchNorm1d(norm_type=norm_type, ret_cls=True)) \
                        or isinstance(m, ModuleHelper.BatchNorm3d(norm_type=norm_type, ret_cls=True)):
                    m.eval()

    @staticmethod
    def clip_grad(net, max_grad=10.):
        """Computes a gradient clipping coefficient based on gradient norm."""
        total_norm = 0
        for p in net.parameters():
            if p.requires_grad:
                modulenorm = p.grad.data.norm()
                total_norm += modulenorm ** 2

        total_norm = math.sqrt(total_norm)

        norm = max_grad / max(total_norm, max_grad)
        for p in net.parameters():
            if p.requires_grad:
                p.grad.mul_(norm)

    @staticmethod
    def gather(runner, outputs, target_device=None, dim=0):
        r"""
        Gathers tensors from different GPUs on a specified device
          (-1 means the CPU).
        """
        if not runner.configer.get('network', 'gather'):
            if target_device is None:
                target_device = list(range(torch.cuda.device_count()))[0]

            return torch_gather(outputs, target_device, dim=dim)

        else:
            return outputs

    @staticmethod
    def dist_avg(runner, data):
        if runner.configer.get('network.distributed'):
            data_list = DistHelper.all_gather(data)
            if isinstance(data, dict):
                return {key:sum([item[key] for item in data_list]) / len(data_list) for key in data}

            if isinstance(data, list):
                return [sum(sub_list) / len(data_list) for sub_list in zip(data_list)]

            return sum(data_list) / len(data_list)

    @staticmethod
    def get_lr(optimizer):

        return [param_group['lr'] for param_group in optimizer.param_groups]

