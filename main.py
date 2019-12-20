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

from runner.runner_selector import RunnerSelector
from lib.runner.controller import Controller
from lib.tools.util.configer import Configer
from lib.tools.util.logger import Logger as Log


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
    parser.add_argument('--config_file', default=None, type=str,
                        dest='config_file', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')

    # ***********  Params for data.  **********
    parser.add_argument('--data_dir', default=None, type=str,
                        dest='data.data_dir', help='The Directory of the data.')
    parser.add_argument('--tag', default=None, type=str,
                        dest='data.tag', help='The Tag of the data.')
    parser.add_argument('--include_val', type=str2bool, nargs='?', default=False,
                        dest='data.include_val', help='Include validation set for final training.')
    parser.add_argument('--drop_last', type=str2bool, nargs='?', default=False,
                        dest='data.drop_last', help='Fix bug for syncbn.')
    parser.add_argument('--workers', default=None, type=int,
                        dest='data.workers', help='The number of workers to load data.')
    parser.add_argument('--train_batch_size', default=None, type=int,
                        dest='train.batch_size', help='The batch size of training.')
    parser.add_argument('--val_batch_size', default=None, type=int,
                        dest='val.batch_size', help='The batch size of validation.')
    parser.add_argument('--test_batch_size', default=None, type=int,
                        dest='test.batch_size', help='The batch size of testing.')

    # ***********  Params for model.  **********
    parser.add_argument('--model_name', default=None, type=str,
                        dest='network.model_name', help='The name of model.')
    parser.add_argument('--checkpoints_root', default=None, type=str,
                        dest='network.checkpoints_root', help='The root dir of model save path.')
    parser.add_argument('--checkpoints_name', default=None, type=str,
                        dest='network.checkpoints_name', help='The name of checkpoint model.')
    parser.add_argument('--backbone', default=None, type=str,
                        dest='network.backbone', help='The base network of model.')
    parser.add_argument('--norm_type', default=None, type=str,
                        dest='network.norm_type', help='The BN type of the network.')
    parser.add_argument('--syncbn',  type=str2bool, nargs='?', default=False,
                        dest='network.syncbn', help='Whether to sync BN.')
    parser.add_argument('--pretrained', type=str, default=None,
                        dest='network.pretrained', help='The path to pretrained model.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network.resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network.resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network.resume_continue', help='Whether to continue training.')
    parser.add_argument('--resume_val', type=str2bool, nargs='?', default=False,
                        dest='network.resume_val', help='Whether to validate during resume.')
    parser.add_argument('--gather', type=str2bool, nargs='?', default=True,
                        dest='network.gather', help='Whether to gather the output of model.')
    parser.add_argument('--dist', type=str2bool, nargs='?', default=False,
                        dest='network.distributed', help='Whether to gather the output of model.')

    # ***********  Params for solver.  **********
    parser.add_argument('--optim_method', default=None, type=str,
                        dest='solver.optim.optim_method', help='The optim method that used.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='solver.lr.base_lr', help='The learning rate.')
    parser.add_argument('--bb_lr_scale', default=1.0, type=float,
                        dest='solver.lr.bb_lr_scale', help='The learning rate.')
    parser.add_argument('--nbb_mult', default=1.0, type=float,
                        dest='solver.lr.nbb_mult', help='The not backbone mult ratio of learning rate.')
    parser.add_argument('--lr_policy', default=None, type=str,
                        dest='solver.lr.lr_policy', help='The policy of lr during training.')
    parser.add_argument('--max_epoch', default=None, type=int,
                        dest='solver.max_epoch', help='The max epoch of training.')
    parser.add_argument('--max_iters', default=None, type=int,
                        dest='solver.max_iters', help='The max iters of training.')
    parser.add_argument('--display_iter', default=None, type=int,
                        dest='solver.display_iter', help='The display iteration of train logs.')
    parser.add_argument('--test_interval', default=None, type=int,
                        dest='solver.test_interval', help='The test interval of validation.')
    parser.add_argument('--save_iters', default=None, type=int,
                        dest='solver.save_iters', help='The saving iters of checkpoint model.')
    parser.add_argument('--save_epoch', default=None, type=int,
                        dest='solver.save_epoch', help='The saving epoch of checkpoint model.')

    # ***********  Params for loss.  **********
    parser.add_argument('--loss_type', default=None, type=str,
                        dest='loss.loss_type', help='The loss type of the network.')

    # ***********  Params for logging.  **********
    parser.add_argument('--log_level', default="info", type=str,
                        dest='logging.log_level', help='To set the level to print to screen.')
    parser.add_argument('--log_format', default="%(asctime)s %(levelname)-7s %(message)s", type=str,
                        dest='logging.log_format', help='To set the format to print to screen.')

    # ***********  Params for test or submission.  **********
    parser.add_argument('--test_dir', default=None, type=str,
                        dest='test.test_dir', help='The test directory of images.')
    parser.add_argument('--out_dir', default='none', type=str,
                        dest='test.out_dir', help='The test out directory of images.')

    # ***********  Params for env.  **********
    parser.add_argument('--seed', default=None, type=int, help='manual seed')
    parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    configer = Configer(args_parser=args)

    if args.seed is not None:
        random.seed(args.seed + args.local_rank)
        torch.manual_seed(args.seed + args.local_rank)

    cudnn.enabled = True
    cudnn.benchmark = args.cudnn

    abs_data_dir = os.path.expanduser(configer.get('data', 'data_dir'))
    configer.update('data.data_dir', abs_data_dir)

    if configer.get('gpu') is not None and not configer.get('network.distributed', default=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in configer.get('gpu'))

    if configer.get('network', 'norm_type') is None:
        configer.update('network.norm_type', 'batchnorm')

    if torch.cuda.device_count() <= 1 or configer.get('network.distributed', default=False):
        configer.update('network.gather', True)

    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add('project_dir', project_dir)

    Log.init(log_level=configer.get('logging', 'log_level'),
             log_format=configer.get('logging', 'log_format'),
             distributed_rank=configer.get('local_rank'))

    Log.info('BN Type is {}.'.format(configer.get('network', 'norm_type')))
    Log.info('Config Dict: {}'.format(json.dumps(configer.to_dict(), indent=2)))

    runner_selector = RunnerSelector(configer)
    runner = None
    if configer.get('task') == 'pose':
        runner = runner_selector.pose_runner()
    elif configer.get('task') == 'seg':
        runner = runner_selector.seg_runner()
    elif configer.get('task') == 'det':
        runner = runner_selector.det_runner()
    elif configer.get('task') == 'cls':
        runner = runner_selector.cls_runner()
    elif configer.get('task') == 'gan':
        runner = runner_selector.gan_runner()
    else:
        Log.error('Task: {} is not valid.'.format(configer.get('task')))
        exit(1)
    if configer.get('phase') == 'train':
        if configer.get('network', 'resume') is None or not configer.get('network.resume_continue'):
            Controller.init(runner)

        Controller.train(runner)
    elif configer.get('phase') == 'test' and configer.get('network', 'resume') is not None:
        Controller.test(runner)
    else:
        Log.error('Phase: {} is not valid.'.format(configer.get('phase')))
        exit(1)
