#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON=python

export PYTHONPATH="/home/donny/Projects/PyTorchCV-SemSeg":$PYTHONPATH

cd ../../../

DATA_DIR="/home/donny/DataSet/Cifar10"
MODEL_NAME="vgg19"
LOSS_TYPE="fc_ce_loss"
CHECKPOINTS_NAME="fc_vgg19_cifar10_cls"
PRETRAINED_MODEL="./pretrained_model/resnet101-imagenet.pth"
MAX_EPOCH=400

LOG_FILE="./log/cityscape/${CHECKPOINTS_NAME}.log"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cls/cifar/fc_vgg19_cifar10_cls.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --model_name ${MODEL_NAME} --gpu 0 1 2 3  \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_epoch ${MAX_EPOCH} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cls/cifar/fc_vgg19_cifar10_cls.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --model_name ${MODEL_NAME} --max_epoch ${MAX_EPOCH} \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} \
                       --resume_continue y --resume ./checkpoints/cityscape/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  >> ${LOG_FILE} 2>&1

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cls/cifar/fc_vgg19_cifar10_cls.json \
                       --phase debug --gpu 0 --log_to_file n  > ${LOG_FILE} 2>&1

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cityscape/fs_deeplabv3_cityscape_seg.json \
                       --model_name ${MODEL_NAME} \
                       --phase test --gpu 0 --resume ./checkpoints/cityscape/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --log_to_file n  >> ${LOG_FILE} 2>&1
  cd val/scripts
  ${PYTHON} -u cityscape_evaluator.py --hypes_file ../../hypes/cityscape/fs_deeplabv3_cityscape_seg.json \
                                      --gt_dir ${DATA_DIR}/val/image \
                                      --pred_dir ../results/cityscape/test_dir/${MODEL_NAME}/label >> ${LOG_FILE} 2>&1
else
  echo "$1"x" is invalid..."
fi