#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'


ORI_IMG_DIR='/home/donny/DataSet/ImageNet/ILSVRC2015/Data/CLS-LOC'
SAVE_DIR='/home/donny/DataSet/ImageNet'

cp valprep.sh $ORI_IMG_DIR'/val'
cd $ORI_IMG_DIR'/val'
sh valprep.sh
cd -

python imagenet_cls_generator.py --ori_img_dir $ORI_IMG_DIR \
                                 --save_dir $SAVE_DIR