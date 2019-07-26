#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="python"

export PYTHONPATH="/home/donny/Projects/TorchCV":${PYTHONPATH}

cd ../../../

DATA_DIR="/home/donny/DataSet/Cifar10"

MODEL_NAME="vgg19"
CHECKPOINTS_NAME="ic_vgg19_cifar10_cls"$2
PRETRAINED_MODEL="./pretrained_models/vgg19-imagenet.pth"

CONFIG_FILE='configs/cls/cifar/ic_vgg19_cifar_cls.json'
MAX_EPOCH=360
LOSS_TYPE="ic_ce_loss"

LOG_DIR="./log/cls/cifar/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi


if [[ "$1"x == "train"x ]]; then
  ${PYTHON} -u main.py --hypes ${CONFIG_FILE} --phase train --gathered n --loss_balance y --model_name ${MODEL_NAME} \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_epoch ${MAX_EPOCH} --gpu 0 1 2 3 \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee ${LOG_FILE}

elif [[ "$1"x == "resume"x ]]; then
  ${PYTHON} -u main.py --hypes ${CONFIG_FILE} --phase train --loss_balance y \
                       --model_name ${MODEL_NAME} --gpu 0 1 2 3 --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} --max_iters ${MAX_EPOCH} \
                       --resume_continue y --resume ./checkpoints/cls/cifar/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee -a ${LOG_FILE}

elif [[ "$1"x == "test"x ]]; then
  ${PYTHON} -u main.py --hypes ${CONFIG_FILE} --phase test --gpu 0 1 2 3 --gathered n \
                       --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/cls/cifar/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/test --out_dir test  2>&1 | tee -a ${LOG_FILE}

else
  echo "$1"x" is invalid..."
fi
