#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some methods used by main methods.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from utils.helpers.file_helper import FileHelper
from utils.tools.logger import Logger as Log


class Controller(object):

    @staticmethod
    def init(runner):
        runner.runner_state['iters'] = 0
        runner.runner_state['last_iters'] = 0
        runner.runner_state['epoch'] = 0
        runner.runner_state['last_epoch'] = 0
        runner.runner_state['performance'] = 0
        runner.runner_state['val_loss'] = 0
        runner.runner_state['max_performance'] = 0
        runner.runner_state['min_val_loss'] = 0

        if not runner.configer.exists('network', 'bn_type'):
            runner.configer.add(['network', 'bn_type'], 'torchbn')

        if len(runner.configer.get('gpu')) == 1:
            runner.configer.update(['network', 'bn_type'], 'torchbn')

        Log.info('BN Type is {}.'.format(runner.configer.get('network', 'bn_type')))

    @staticmethod
    def train(runner):
        Log.info('Training start...')
        if runner.configer.get('network', 'resume') is not None and runner.configer.get('network', 'resume_val'):
            runner.val()

        if runner.configer.get('lr', 'metric') == 'epoch':
            while runner.runner_state['epoch'] < runner.configer.get('solver', 'max_epoch'):
                runner.train()
                if runner.runner_state['epoch'] == runner.configer.get('solver', 'max_epoch'):
                    runner.val()
                    break
        else:
            while runner.runner_state['iters'] < runner.configer.get('solver', 'max_iters'):
                runner.train()
                if runner.runner_state['iters'] == runner.configer.get('solver', 'max_iters'):
                    runner.val()
                    break

        Log.info('Training end...')

    @staticmethod
    def debug(runner):
        Log.info('Debugging start..')
        base_dir = os.path.join(runner.configer.get('project_dir'), 'vis/results',
                                runner.configer.get('task'), runner.configer.get('network', 'model_name'))

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        runner.debug(base_dir)
        Log.info('Debugging end...')

    @staticmethod
    def test(runner):
        Log.info('Testing start...')
        base_dir = os.path.join(runner.configer.get('project_dir'),
                                'val/results', runner.configer.get('task'),
                                runner.configer.get('network', 'model_name'))

        test_img = runner.configer.get('test_img')
        test_dir = runner.configer.get('test_dir')
        if test_img is None and test_dir is None:
            Log.error('test_img & test_dir not exists.')
            exit(1)

        if test_img is not None and test_dir is not None:
            Log.error('Either test_img or test_dir.')
            exit(1)

        if test_img is not None:
            base_dir = os.path.join(base_dir, 'test_img')
            filename = test_img.rstrip().split('/')[-1]
            label_path = os.path.join(base_dir, 'label', '{}.png'.format('.'.join(filename.split('.')[:-1])))
            raw_path = os.path.join(base_dir, 'raw', filename)
            vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
            FileHelper.make_dirs(label_path, is_file=True)
            FileHelper.make_dirs(raw_path, is_file=True)
            FileHelper.make_dirs(vis_path, is_file=True)

            runner.test_img(test_img, label_path, vis_path, raw_path)

        else:
            base_dir = os.path.join(base_dir, 'test_dir', test_dir.rstrip('/').split('/')[-1])
            FileHelper.make_dirs(base_dir)

            for filename in FileHelper.list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                label_path = os.path.join(base_dir, 'label', '{}.png'.format('.'.join(filename.split('.')[:-1])))
                raw_path = os.path.join(base_dir, 'raw', filename)
                vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
                FileHelper.make_dirs(label_path, is_file=True)
                FileHelper.make_dirs(raw_path, is_file=True)
                FileHelper.make_dirs(vis_path, is_file=True)

                runner.test_img(image_path, label_path, vis_path, raw_path)

        Log.info('Testing end...')
