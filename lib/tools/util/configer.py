#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Configer class for all hyper parameters.


import sys
import json
import os
import argparse

from pyhocon import ConfigFactory

from lib.tools.util.logger import Logger as Log


class Configer(object):

    def __init__(self, args_parser=None, config_file=None, config_dict=None, valid_flag=None):
        self.params_root = None
        if config_dict is not None:
            assert config_file is None
            self.params_root = ConfigFactory.from_dict(config_dict)

        elif config_file is not None:
            if not os.path.exists(config_file):
                Log.error('Json Path:{} not exists!'.format(config_file))
                exit(1)

            self.params_root = ConfigFactory.parse_file(config_file)

        elif 'config_file' in args_parser and args_parser.config_file is not None:
            if not os.path.exists(args_parser.config_file):
                Log.error('Json Path:{} not exists!'.format(args_parser.config_file))
                exit(1)

            self.params_root = ConfigFactory.parse_file(args_parser.config_file)

        else:
            Log.warn('Base settings not set!')
            self.params_root = ConfigFactory.from_dict({})

        if args_parser is not None:
            for key, value in args_parser.__dict__.items():
                if valid_flag is not None and key.split('.')[0] != valid_flag:
                    continue

                if key not in self.params_root:
                    self.add(key, value)
                elif value is not None:
                    self.update(key, value)

    def _get_caller(self):
        filename = os.path.basename(sys._getframe().f_back.f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_back.f_lineno
        prefix = '{}, {}'.format(filename, lineno)
        return prefix

    def get(self, *key, **kwargs):
        key = '.'.join(key)
        if key in self.params_root or 'default' in kwargs:
            return self.params_root.get(key, **kwargs)

        else:
            Log.error('{} KeyError: {}.'.format(self._get_caller(), key))
            exit(1)

    def add(self, key, value):
        if key in self.params_root:
            Log.error('{} Key: {} existed!!!'.format(self._get_caller(), key))
            exit(1)

        self.params_root.put(key, value)

    def update(self, key, value, append=False):
        if key not in self.params_root:
            Log.error('{} Key: {} not existed!!!'.format(self._get_caller(), key))
            exit(1)

        self.params_root.put(key, value, append)

    def resume(self, config_dict):
        self.params_root = ConfigFactory.from_dict(config_dict)

    def to_dict(self):
        return self.params_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='', type=str,
                        dest='configs', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of Pose Estimator.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network.resume', help='The path of pretrained model.')
    parser.add_argument('--train_dir', default=None, type=str,
                        dest='data.train_dir', help='The path of train data.')

    args_parser = parser.parse_args()

    configer = Configer(args_parser=args_parser)

    configer.add('project_dir', 'root')
    configer.update('project_dir', 'root1')

    print (configer.get('project_dir'))
    print (configer.get('network.resume'))
    print (configer.get('logging', 'log_file'))
    print(configer.get('data', 'train_dir'))
