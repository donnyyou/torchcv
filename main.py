#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Main Scripts for computer vision.


import os
import json
import time
import random
import argparse
import torch
import torch.backends.cudnn as cudnn

from methods.method_selector import MethodSelector
from methods.tools.controller import Controller
from utils.tools.configer import Configer
from utils.tools.logger import Logger as Log


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', default=None, type=str,
                        dest='hypes', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')

    # ***********  Params for data.  **********
    parser.add_argument('--data_dir', default=None, type=str,
                        dest='data:data_dir', help='The Directory of the data.')
    parser.add_argument('--tag', default=None, type=str,
                        dest='data:tag', help='The Tag of the data.')
    parser.add_argument('--include_val', type=str2bool, nargs='?', default=False,
                        dest='data:include_val', help='Include validation set for final training.')
    parser.add_argument('--drop_last', type=str2bool, nargs='?', default=False,
                        dest='data:drop_last', help='Fix bug for syncbn.')
    parser.add_argument('--workers', default=None, type=int,
                        dest='data:workers', help='The number of workers to load data.')
    parser.add_argument('--train_batch_size', default=None, type=int,
                        dest='train:batch_size', help='The batch size of training.')
    parser.add_argument('--val_batch_size', default=None, type=int,
                        dest='val:batch_size', help='The batch size of validation.')

    # ***********  Params for model.  **********
    parser.add_argument('--model_name', default=None, type=str,
                        dest='network:model_name', help='The name of model.')
    parser.add_argument('--checkpoints_root', default=None, type=str,
                        dest='network:checkpoints_root', help='The root dir of model save path.')
    parser.add_argument('--checkpoints_name', default=None, type=str,
                        dest='network:checkpoints_name', help='The name of checkpoint model.')
    parser.add_argument('--backbone', default=None, type=str,
                        dest='network:backbone', help='The base network of model.')
    parser.add_argument('--norm_type', default=None, type=str,
                        dest='network:norm_type', help='The BN type of the network.')
    parser.add_argument('--multi_grid', default=None, nargs='+', type=int,
                        dest='network:multi_grid', help='The multi_grid for resnet backbone.')
    parser.add_argument('--pretrained', type=str, default=None,
                        dest='network:pretrained', help='The path to pretrained model.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network:resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network:resume_continue', help='Whether to continue training.')
    parser.add_argument('--resume_val', type=str2bool, nargs='?', default=True,
                        dest='network:resume_val', help='Whether to validate during resume.')
    parser.add_argument('--gathered', type=str2bool, nargs='?', default=True,
                        dest='network:gathered', help='Whether to gather the output of model.')
    parser.add_argument('--loss_balance', type=str2bool, nargs='?', default=False,
                        dest='network:loss_balance', help='Whether to balance GPU usage.')

    # ***********  Params for solver.  **********
    parser.add_argument('--optim_method', default=None, type=str,
                        dest='solver:optim:optim_method', help='The optim method that used.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='solver:lr:base_lr', help='The learning rate.')
    parser.add_argument('--nbb_mult', default=1.0, type=float,
                        dest='solver:lr:nbb_mult', help='The not backbone mult ratio of learning rate.')
    parser.add_argument('--lr_policy', default=None, type=str,
                        dest='solver:lr:lr_policy', help='The policy of lr during training.')
    parser.add_argument('--max_epoch', default=None, type=int,
                        dest='solver:max_epoch', help='The max epoch of training.')
    parser.add_argument('--max_iters', default=None, type=int,
                        dest='solver:max_iters', help='The max iters of training.')
    parser.add_argument('--display_iter', default=None, type=int,
                        dest='solver:display_iter', help='The display iteration of train logs.')
    parser.add_argument('--test_interval', default=None, type=int,
                        dest='solver:test_interval', help='The test interval of validation.')
    parser.add_argument('--save_iters', default=None, type=int,
                        dest='solver:save_iters', help='The saving iters of checkpoint model.')
    parser.add_argument('--save_epoch', default=None, type=int,
                        dest='solver:save_epoch', help='The saving epoch of checkpoint model.')

    # ***********  Params for loss.  **********
    parser.add_argument('--loss_type', default=None, type=str,
                        dest='loss:loss_type', help='The loss type of the network.')

    # ***********  Params for logging.  **********
    parser.add_argument('--logfile_level', default=None, type=str,
                        dest='logging:logfile_level', help='To set the log level to files.')
    parser.add_argument('--stdout_level', default=None, type=str,
                        dest='logging:stdout_level', help='To set the level to print to screen.')
    parser.add_argument('--log_file', default=None, type=str,
                        dest='logging:log_file', help='The path of log files.')
    parser.add_argument('--rewrite', type=str2bool, nargs='?', default=True,
                        dest='logging:rewrite', help='Whether to rewrite files.')
    parser.add_argument('--log_to_file', type=str2bool, nargs='?', default=True,
                        dest='logging:log_to_file', help='Whether to write logging into files.')

    # ***********  Params for test or submission.  **********
    parser.add_argument('--test_img', default=None, type=str,
                        dest='test:img_path', help='The test path of image.')
    parser.add_argument('--test_dir', default=None, type=str,
                        dest='test:data_dir', help='The test directory of images.')
    parser.add_argument('--out_dir', default='none', type=str,
                        dest='test:out_dir', help='The test out directory of images.')

    # ***********  Params for env.  **********
    parser.add_argument('--seed', default=None, type=int, help='manual seed')
    parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')

    args_parser = parser.parse_args()

    if args_parser.seed is not None:
        random.seed(args_parser.seed)
        torch.manual_seed(args_parser.seed)

    cudnn.enabled = True
    cudnn.benchmark = args_parser.cudnn

    configer = Configer(args_parser=args_parser)
    abs_data_dir = os.path.expanduser(configer.get('data', 'data_dir'))
    configer.update(['data', 'data_dir'], abs_data_dir)

    if configer.get('gpu') is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in configer.get('gpu'))

    if configer.get('network', 'norm_type') is None:
        configer.update(['network', 'norm_type'], 'batchnorm')

    if configer.get('phase') == 'train':
        assert len(configer.get('gpu')) > 1 or 'sync' not in configer.get('network', 'norm_type')

    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add(['project_dir'], project_dir)

    if configer.get('logging', 'log_to_file'):
        log_file = configer.get('logging', 'log_file')
        new_log_file = '{}_{}'.format(log_file, time.strftime("%Y-%m-%d_%X", time.localtime()))
        configer.update(['logging', 'log_file'], new_log_file)
    else:
        configer.update(['logging', 'logfile_level'], None)

    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'))

    Log.info('BN Type is {}.'.format(configer.get('network', 'norm_type')))
    Log.info('Config Dict: {}'.format(json.dumps(configer.to_dict(), indent=2)))
    method_selector = MethodSelector(configer)
    runner = None
    if configer.get('task') == 'pose':
        runner = method_selector.select_pose_method()
    elif configer.get('task') == 'seg':
        runner = method_selector.select_seg_method()
    elif configer.get('task') == 'det':
        runner = method_selector.select_det_method()
    elif configer.get('task') == 'cls':
        runner = method_selector.select_cls_method()
    elif configer.get('task') == 'gan':
        runner = method_selector.select_gan_method()
    else:
        Log.error('Task: {} is not valid.'.format(configer.get('task')))
        exit(1)

    if configer.get('phase') == 'train':
        if configer.get('network', 'resume') is None:
            Controller.init(runner)

        Controller.train(runner)
    elif configer.get('phase') == 'debug':
        Controller.debug(runner)
    elif configer.get('phase') == 'test' and configer.get('network', 'resume') is not None:
        Controller.test(runner)
    else:
        Log.error('Phase: {} is not valid.'.format(configer.get('phase')))
        exit(1)
