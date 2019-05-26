#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/torchcv'


ROOT_DIR='/home/donny/DataSet/voc/VOCdevkit'
DATA_SET='VOC07+12'
SAVE_DIR='/home/donny/DataSet/'${DATA_SET}'_DET'


python voc_det_generator.py --root_dir $ROOT_DIR \
                            --save_dir $SAVE_DIR \
                            --dataset $DATA_SET
