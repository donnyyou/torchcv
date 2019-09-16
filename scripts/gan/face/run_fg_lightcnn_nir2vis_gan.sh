#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="python"

WORK_DIR=$(cd $(dirname $0)/../../../;pwd)
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
cd ${WORK_DIR}

DATA_DIR="/home/donny/DataSet/Face"

BACKBONE="deepbase_resnet101_dilated8"
MODEL_NAME="facegan"
CHECKPOINTS_NAME="fg_lightcnn_face_gan"$2
PRETRAINED_MODEL="./pretrained_models/3x3resnet101-imagenet.pth"

CONFIG_FILE='configs/gan/face/fg_lightcnn_nir2vis_gan.conf'
MAX_ITERS=40000
LOSS_TYPE="fs_auxce_loss"

LOG_DIR="./log/gan/face/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi


if [[ "$1"x == "train"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --drop_last y --phase train --loss_balance y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee ${LOG_FILE}

elif [[ "$1"x == "resume"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --drop_last y --phase train --gathered n --loss_balance y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --resume_continue y --resume ./checkpoints/seg/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee -a ${LOG_FILE}

elif [[ "$1"x == "test"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --phase test --gpu 0 1 2 3 --gathered n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/gan/face/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/test --out_dir test >> ${LOG_FILE}  2>&1 | tee -a ${LOG_FILE}

else
  echo "$1"x" is invalid..."
fi
