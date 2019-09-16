#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Zhen Zhu(zzhu@hust.edu.cn)
# Generate train & val data.


ORI_ROOT_DIR='/home/zhuz/.encoding/data/VOCaug/dataset'
SAVE_DIR='/home/zhuz/datasets/PascalAUG'


python pascal_aug_generator.py --ori_root_dir $ORI_ROOT_DIR \
                                   --save_dir $SAVE_DIR
