#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


ORI_ROOT_DIR='/home/donny/DataSet/ADE20K'
SAVE_DIR='/home/donny/DataSet/ADE20K'


python ade20k_seg_generator.py --ori_root_dir $ORI_ROOT_DIR \
                               --save_dir $SAVE_DIR