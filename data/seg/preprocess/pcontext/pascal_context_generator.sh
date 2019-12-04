#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Zhen Zhu(zzhu@hust.edu.cn)
# Generate train & val data.


ORI_ROOT_DIR='/home/zhuz/.encoding/data/VOCdevkit/VOC2010'
SAVE_DIR='/home/zhuz/datasets/PascalContext'


python pascal_context_generator.py --ori_root_dir $ORI_ROOT_DIR \
                                   --save_dir $SAVE_DIR
