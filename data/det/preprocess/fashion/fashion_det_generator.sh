#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'


ORI_IMG_DIR='/home/donny/DataSet/DeepFashion/img'
ORI_TYPE_FILE='/home/donny/DataSet/DeepFashion/Anno/list_joints.txt'
ORI_BBOX_FILE='/home/donny/DataSet/DeepFashion/Anno/list_bbox.txt'
SAVE_DIR='/home/donny/DataSet/Fashion'


python fashion_det_generator.py --ori_img_dir $ORI_IMG_DIR \
                                --ori_type_file $ORI_TYPE_FILE \
                                --ori_bbox_file $ORI_BBOX_FILE \
                                --save_dir $SAVE_DIR \
                                --val_interval 10
