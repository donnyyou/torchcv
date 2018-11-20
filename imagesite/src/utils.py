#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import config


def list_dir(dir_path):
    sub_parts = dir_path.split('/')
    if sub_parts[0] == 'dataset':
        absolute_dir = os.path.join(config.DATASET_ROOT, '/'.join(sub_parts[1:]))
    else:
        absolute_dir = os.path.join(config.PROJECT_ROOT, '/'.join(sub_parts[1:]))

    result_json = dict()
    result_json['items'] = list()

    is_img_folder = True

    for item in os.listdir(absolute_dir):
        item_dict = dict()
        item_dict['dir_name'] = item
        item_dict['url'] = '{}/{}/{}'.format(config.STATIC_SERVICE, dir_path, item)
        if is_img_file(item):
            item_dict['type'] = 'image'
            shotname, extension = os.path.splitext(item)
            small_name = '{}_100x100{}'.format(shotname, extension)
            item_dict['smallurl'] = '{}/{}/{}'.format(config.STATIC_SERVICE, dir_path, small_name)
        else:
            if os.path.isdir(os.path.join(absolute_dir, item)):
                item_dict['type'] = 'dir'
            else:
                item_dict['type'] = 'file'

            is_img_folder= False

        item_dict['path'] = '{}/{}'.format(dir_path, item)
        result_json['items'].append(item_dict)

    result_json['is_img_folder'] = is_img_folder

    return result_json


def list_jsons():
    result_json = dict()
    task_list = ['cls', 'seg', 'pose', 'det']
    for task in task_list:
        result_json[task] = dict()
        for dataset in os.listdir(os.path.join(config.PROJECT_ROOT, 'hypes', task)):
            result_json[task][dataset] = list()
            for json_file in os.listdir(os.path.join(config.PROJECT_ROOT, 'hypes', task, dataset)):
                result_json[task][dataset].append(json_file)

    return result_json


def list_params(json_name):
    json_name_items = json_name.rstrip().split('_')
    task_type = json_name_items[-1]
    dataset = json_name_items[-2]
    result_json = dict()
    result_json['phase'] = ['train', 'test', 'submission']
    result_json['resume'] = list()
    for checkpoint in os.listdir(os.path.join(config.PROJECT_ROOT, 'checkpoints', task_type, dataset)):
        result_json['resume'].append(checkpoint)

    result_json['base_lr'] = 0.0001
    result_json['train_batch_size'] = 1
    result_json['val_batch_size'] = 1
    result_json['pretrained'] = ['false', 'true']
    result_json['test_dir'] = 'None'
    result_json['test_img'] = 'None'

    return result_json


def is_img_file(file_name):
    return any(file_name.endswith(extension) for extension in config.IMG_EXTENSIONS)


if __name__ == "__main__":
    print list_dir('dataset/road_marker_5000')
