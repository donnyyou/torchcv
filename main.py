#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)
# Main Scripts for computer vision.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse

from methods.method_selector import MethodSelector
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
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')

    # ***********  Params for data.  **********
    parser.add_argument('--data_dir', default=None, type=str,
                        dest='data:data_dir', help='The Directory of the data.')
    parser.add_argument('--workers', default=None, type=int,
                        dest='data:workers', help='The number of workers to load data.')
    parser.add_argument('--train_batch_size', default=None, type=int,
                        dest='train:batch_size', help='The batch size of training.')
    parser.add_argument('--val_batch_size', default=None, type=int,
                        dest='val:batch_size', help='The batch size of validation.')

    # ***********  Params for checkpoint.  **********
    parser.add_argument('--checkpoints_root', default=None, type=str,
                        dest='checkpoints:checkpoints_root', help='The root dir of model save path.')
    parser.add_argument('--checkpoints_name', default=None, type=str,
                        dest='checkpoints:checkpoints_name', help='The name of checkpoint model.')
    parser.add_argument('--save_iters', default=None, type=int,
                        dest='checkpoints:save_iters', help='The saving iters of checkpoint model.')
    parser.add_argument('--save_epoch', default=None, type=int,
                        dest='checkpoints:save_epoch', help='The saving epoch of checkpoint model.')

    # ***********  Params for model.  **********
    parser.add_argument('--model_name', default=None, type=str,
                        dest='network:model_name', help='The name of model.')
    parser.add_argument('--backbone', default=None, type=str,
                        dest='network:backbone', help='The base network of model.')
    parser.add_argument('--pretrained', type=str, default=None,
                        dest='network:pretrained', help='The path to pretrained model.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of checkpoints.')
    parser.add_argument('--resume_level', default='full', type=str,
                        dest='network:resume_level', help='The resume level of networks.')
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
                        dest='optim:optim_method', help='The optim method that used.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='lr:base_lr', help='The learning rate.')
    parser.add_argument('--lr_policy', default=None, type=str,
                        dest='lr:lr_policy', help='The policy of lr during training.')

    # ***********  Params for display.  **********
    parser.add_argument('--max_epoch', default=None, type=int,
                        dest='solver:max_epoch', help='The max epoch of training.')
    parser.add_argument('--display_iter', default=None, type=int,
                        dest='solver:display_iter', help='The display iteration of train logs.')
    parser.add_argument('--test_interval', default=None, type=int,
                        dest='solver:test_interval', help='The test interval of validation.')

    # ***********  Params for logging.  **********
    parser.add_argument('--logfile_level', default=None, type=str,
                        dest='logging:logfile_level', help='To set the log level to files.')
    parser.add_argument('--stdout_level', default=None, type=str,
                        dest='logging:stdout_level', help='To set the level to print to screen.')
    parser.add_argument('--log_file', default=None, type=str,
                        dest='logging:log_file', help='The path of log files.')

    # ***********  Params for test or submission.  **********
    parser.add_argument('--test_img', default=None, type=str,
                        dest='test_img', help='The test path of image.')
    parser.add_argument('--test_dir', default=None, type=str,
                        dest='test_dir', help='The test directory of images.')

    args_parser = parser.parse_args()

    configer = Configer(args_parser=args_parser)
    abs_data_dir = os.path.expanduser(configer.get('data', 'data_dir'))
    configer.update_value(['data', 'data_dir'], abs_data_dir)

    if configer.get('gpu') is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in configer.get('gpu'))

    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add_key_value(['project_dir'], project_dir)

    log_file = configer.get('logging', 'log_file')
    new_log_file = '{}_{}'.format(log_file, time.strftime("%Y-%m-%d_%X", time.localtime()))
    configer.update_value(['logging', 'log_file'], new_log_file)

    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'))

    method_selector = MethodSelector(configer)
    model = None
    if configer.get('task') == 'pose':
        model = method_selector.select_pose_method()
    elif configer.get('task') == 'seg':
        model = method_selector.select_seg_method()
    elif configer.get('task') == 'det':
        model = method_selector.select_det_method()
    elif configer.get('task') == 'cls':
        model = method_selector.select_cls_method()
    elif configer.get('task') == 'multitask':
        model = method_selector.select_multitask_method()
    else:
        Log.error('Task: {} is not valid.'.format(configer.get('task')))
        exit(1)

    if configer.get('phase') == 'train':
        model.train()
    elif configer.get('phase') == 'debug':
        model.debug()
    elif configer.get('phase') == 'test' and configer.get('network', 'resume') is not None:
        model.test()
    else:
        Log.error('Phase: {} is not valid.'.format(configer.get('phase')))
        exit(1)
