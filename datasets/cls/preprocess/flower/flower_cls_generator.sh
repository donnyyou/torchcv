#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'


ORI_IMG_DIR='/home/donny/Downloads/jpg'
SAVE_DIR='/data/DataSet/Flower'


python flower_cls_generator.py --ori_img_dir $ORI_IMG_DIR \
                               --save_dir $SAVE_DIR \
                               --val_interval 10