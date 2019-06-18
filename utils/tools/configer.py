#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Configer class for all hyper parameters.


import sys
import json
import os
import argparse

from utils.tools.logger import Logger as Log


class Configer(object):

    def __init__(self, args_parser=None, configs=None, config_dict=None):
        self.params_root = None
        if config_dict is not None and configs is None:
            self.params_root = config_dict

        if config_dict is None and configs is not None:
            if not os.path.exists(configs):
                Log.error('Json Path:{} not exists!'.format(configs))
                exit(1)

            json_stream = open(configs, 'r')
            self.params_root = json.load(json_stream)
            json_stream.close()

        if args_parser is not None and self.params_root is None:
            assert configs is None and config_dict is None
            if not os.path.exists(args_parser.configs):
                Log.error('Json Path:{} not exists!'.format(args_parser.configs))
                exit(1)

            json_stream = open(args_parser.configs, 'r')
            self.params_root = json.load(json_stream)
            json_stream.close()

        assert self.params_root is not None
        if args_parser is not None:
            for key, value in args_parser.__dict__.items():
                if not self.exists(*key.split(':')):
                    self.add(key.split(':'), value)
                elif value is not None:
                    self.update(key.split(':'), value)

    def _get_caller(self):
        filename = os.path.basename(sys._getframe().f_back.f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_back.f_lineno
        prefix = '{}, {}'.format(filename, lineno)
        return prefix

    def get(self, *key):
        if len(key) == 0:
            return self.params_root

        elif len(key) == 1:
            if key[0] in self.params_root:
                return self.params_root[key[0]]
            else:
                Log.error('{} KeyError: {}.'.format(self._get_caller(), key))
                exit(1)

        elif len(key) == 2:
            if key[0] in self.params_root and key[1] in self.params_root[key[0]]:
                return self.params_root[key[0]][key[1]]
            else:
                Log.error('{} KeyError: {}.'.format(self._get_caller(), key))
                exit(1)

        else:
            Log.error('{} KeyError: {}.'.format(self._get_caller(), key))
            exit(1)

    def exists(self, *key):
        if len(key) == 1 and key[0] in self.params_root:
            return True

        if len(key) == 2 and (key[0] in self.params_root and key[1] in self.params_root[key[0]]):
            return True

        if len(key) == 3 and (key[0] in self.params_root and key[1] in self.params_root[key[0]])\
                and key[2] in self.params_root[key[0]][key[1]]:
            return True

        if len(key) == 4 and (key[0] in self.params_root and key[1] in self.params_root[key[0]])\
                and key[2] in self.params_root[key[0]][key[1]] and key[3] in self.params_root[key[0]][key[1]][key[2]]:
            return True

        return False

    def add(self, key_tuple, value):
        if self.exists(*key_tuple):
            Log.error('{} Key: {} existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

        if len(key_tuple) == 1:
            self.params_root[key_tuple[0]] = value

        elif len(key_tuple) == 2:
            if key_tuple[0] not in self.params_root:
                self.params_root[key_tuple[0]] = dict()

            self.params_root[key_tuple[0]][key_tuple[1]] = value

        elif len(key_tuple) == 3:
            if key_tuple[0] not in self.params_root:
                self.params_root[key_tuple[0]] = dict()

            if key_tuple[1] not in self.params_root[key_tuple[0]]:
                self.params_root[key_tuple[0]][key_tuple[1]] = dict()

            self.params_root[key_tuple[0]][key_tuple[1]][key_tuple[2]] = value

        elif len(key_tuple) == 4:
            if key_tuple[0] not in self.params_root:
                self.params_root[key_tuple[0]] = dict()

            if key_tuple[1] not in self.params_root[key_tuple[0]]:
                self.params_root[key_tuple[0]][key_tuple[1]] = dict()

            if key_tuple[2] not in self.params_root[key_tuple[0]][key_tuple[1]]:
                self.params_root[key_tuple[0]][key_tuple[1]][key_tuple[2]] = dict()

            self.params_root[key_tuple[0]][key_tuple[1]][key_tuple[2]][key_tuple[3]] = value

        else:
            Log.error('{} KeyError: {}.'.format(self._get_caller(), key_tuple))
            exit(1)

    def update(self, key_tuple, value):
        if not self.exists(*key_tuple):
            Log.error('{} Key: {} not existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

        if len(key_tuple) == 1:
            assert not isinstance(self.params_root[key_tuple[0]], dict)
            self.params_root[key_tuple[0]] = value

        elif len(key_tuple) == 2:
            assert not isinstance(self.params_root[key_tuple[0]][key_tuple[1]], dict)
            self.params_root[key_tuple[0]][key_tuple[1]] = value

        elif len(key_tuple) == 3:
            assert not isinstance(self.params_root[key_tuple[0]][key_tuple[1]][key_tuple[2]], dict)
            self.params_root[key_tuple[0]][key_tuple[1]][key_tuple[2]] = value

        elif len(key_tuple) == 4:
            assert not isinstance(self.params_root[key_tuple[0]][key_tuple[1]][key_tuple[2]][key_tuple[3]], dict)
            self.params_root[key_tuple[0]][key_tuple[1]][key_tuple[2]][key_tuple[3]] = value

        else:
            Log.error('{} Key: {} not existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

    def resume(self, config_dict):
        self.params_root = config_dict

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
                        dest='network:resume', help='The path of pretrained model.')
    parser.add_argument('--train_dir', default=None, type=str,
                        dest='data:train_dir', help='The path of train data.')

    args_parser = parser.parse_args()

    configer = Configer(args_parser=args_parser)

    configer.add(('project_dir',), 'root')
    configer.update(('project_dir',), 'root1')

    print (configer.get('project_dir'))
    print (configer.get('network', 'resume'))
    print (configer.get('logging', 'log_file'))
    print(configer.get('data', 'train_dir'))
