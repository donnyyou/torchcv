#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="python"

WORK_DIR=$(cd $(dirname $0)/../../../;pwd)
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
cd ${WORK_DIR}

DATA_DIR="/home/donny/DataSet/VOC07_DET"
MODEL_NAME="faster_rcnn"
LOSS_TYPE="fasterrcnn_loss"
CHECKPOINTS_NAME="fr_vgg16_voc_det"$2
PRETRAINED_MODEL="./pretrained_models/faster_rcnn_vgg16_caffe_pretrained.pth"
CONFIG_FILE='configs/det/voc/fasterrcnn_vgg16_voc_det.conf'

LOG_DIR="./log/det/voc/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi

if [[ "$1"x == "train"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --phase train --gpu 2 --cudnn n \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee ${LOG_FILE}

elif [[ "$1"x == "resume"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --phase train --gpu 0 --cudnn n \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --model_name ${MODEL_NAME} \
                       --resume_continue y --resume ./checkpoints/det/voc/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee -a ${LOG_FILE}

elif [[ "$1"x == "val"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --phase test --model_name ${MODEL_NAME} \
                       --phase test --gpu 0 1 2 3 --resume ./checkpoints/det/voc/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --out_dir val  2>&1 | tee -a ${LOG_FILE}
  cd metric/det/
  ${PYTHON} -u voc_evaluator.py --config_file "../../"${CONFIG_FILE} \
                                --json_dir ../../results/det/voc/${CHECKPOINTS_NAME}/val/json \
                                --gt_dir ${DATA_DIR}/VOCdevkit/VOC2007  2>&1 | tee -a "../../"${LOG_FILE}

else
  echo "$1"x" is invalid..."
fi
