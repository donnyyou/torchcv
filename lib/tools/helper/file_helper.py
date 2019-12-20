#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Repackage some file operations.


import os


class FileHelper(object):

    @staticmethod
    def make_dirs(dir_path, is_file=False):
        dir_path = os.path.expanduser(dir_path)
        dir_name = FileHelper.dir_name(dir_path) if is_file else dir_path
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def dir_name(file_path):
        return os.path.dirname(file_path)

    @staticmethod
    def abs_path(file_path):
        return os.path.abspath(file_path)

    @staticmethod
    def shotname(file_name):
        shotname, extension = os.path.splitext(file_name)
        return shotname

    @staticmethod
    def scandir(dir_path, suffix=None):
        for entry in os.scandir(dir_path):
            if not entry.is_file():
                continue
            filename = entry.name
            if suffix is None:
                yield filename
            elif filename.endswith(suffix):
                yield filename

    @staticmethod
    def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
        if not os.path.isfile(filename):
            raise FileNotFoundError(msg_tmpl.format(filename))

    @staticmethod
    def list_dir(dir_name, prefix=''):
        filename_list = list()
        items = os.listdir(os.path.join(dir_name, prefix))
        for item in items:
            fi_d = os.path.join(dir_name, prefix, item)
            if os.path.isdir(fi_d):
                prefix_temp = '{}/{}'.format(prefix, item).lstrip('/')
                filename_list += FileHelper.list_dir(dir_name, prefix_temp)
            else:
                filename_list.append('{}/{}'.format(prefix, item).lstrip('/'))

        return filename_list


if __name__ == "__main__":
    print (FileHelper.list_dir('/home/donny/Projects'))
