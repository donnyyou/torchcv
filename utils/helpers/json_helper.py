#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Repackage some json operations.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from utils.tools.logger import Logger as Log


class JsonHelper(object):

    @staticmethod
    def load_file(json_file):
        if not os.path.exists(json_file):
            Log.error('Json file: {} not exists.'.format(json_file))
            exit(1)

        with open(json_file, 'r') as read_stream:
            json_dict = json.load(read_stream)

        return json_dict

    @staticmethod
    def save_file(json_dict, json_file):
        dir_name = os.path.dirname(json_file)
        if not os.path.exists(dir_name):
            Log.info('Json Dir: {} not exists.'.format(dir_name))
            os.makedirs(dir_name)

        with open(json_file, 'w') as write_stream:
            write_stream.write(json.dumps(json_dict))

    @staticmethod
    def json2xml(json_file, xml_file):
        if not os.path.exists(json_file):
            Log.error('Json file: {} not exists.'.format(json_file))
            exit(1)

        xml_dir_name = os.path.dirname(xml_file)
        if not os.path.exists(xml_dir_name):
            Log.info('Xml Dir: {} not exists.'.format(xml_dir_name))
            os.makedirs(xml_dir_name)

    @staticmethod
    def xml2json(xml_file, json_file):
        if not os.path.exists(xml_file):
            Log.error('Xml file: {} not exists.'.format(xml_file))
            exit(1)

        json_dir_name = os.path.dirname(json_file)
        if not os.path.exists(json_dir_name):
            Log.info('Json Dir: {} not exists.'.format(json_dir_name))
            os.makedirs(json_dir_name)