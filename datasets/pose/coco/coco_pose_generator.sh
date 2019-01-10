#!/usr/bin bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'

INPUT_SIZE=368

COCO_DIR='/home/donny/DataSet/MSCOCO/'
COCO_TRAIN_IMG_DIR=${COCO_DIR}'train2017'
COCO_VAL_IMG_DIR=${COCO_DIR}'/val2017'

COCO_ANNO_DIR=${COCO_DIR}'annotations/'
TRAIN_ANNO_FILE=${COCO_ANNO_DIR}'person_keypoints_train2017.json'
VAL_ANNO_FILE=${COCO_ANNO_DIR}'person_keypoints_val2017.json'

SAVE_DIR='/home/donny/DataSet/COCO_KPTS/'
TRAIN_ROOT_DIR=${SAVE_DIR}'train'
VAL_ROOT_DIR=${SAVE_DIR}'val'


python coco_pose_generator.py --root_dir $TRAIN_ROOT_DIR \
                              --ori_anno_file $TRAIN_ANNO_FILE \
                              --ori_img_dir $COCO_TRAIN_IMG_DIR

python coco_pose_generator.py --root_dir $VAL_ROOT_DIR \
                              --ori_anno_file $VAL_ANNO_FILE \
                              --ori_img_dir $COCO_VAL_IMG_DIR
