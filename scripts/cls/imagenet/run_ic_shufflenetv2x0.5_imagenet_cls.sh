#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="python -u"

WORK_DIR=$(cd $(dirname $0)/../../../; pwd)
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
cd ${WORK_DIR}

DATA_DIR="/home/donny/DataSets/ImageNet"

MODEL_NAME="cls_model"
BACKBONE="shufflenet_v2_x0_5"
CHECKPOINTS_NAME="${BACKBONE}_imagenet_cls"$2

CONFIG_FILE='configs/cls/imagenet/base_small_imagenet_cls.conf'
MAX_ITERS=400000
LOSS_TYPE="soft_ce_loss"

LOG_DIR="./log/cls/imagenet/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [ ! -d ${LOG_DIR} ]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi

NGPUS=4
DIST_PYTHON="${PYTHON} -m torch.distributed.launch --nproc_per_node=${NGPUS}"

if [ -z $1 ]; then
  echo "Phase is None..."

elif [ $1 == train ]; then
  ${DIST_PYTHON} main.py --config_file ${CONFIG_FILE} --phase train --gather y --model_name ${MODEL_NAME} \
                         --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} --dist y \
                         --backbone ${BACKBONE} --checkpoints_name ${CHECKPOINTS_NAME}  2>&1 | tee ${LOG_FILE}

elif [ $1 == resume ]; then
  ${DIST_PYTHON} main.py --config_file ${CONFIG_FILE} --phase train --gather y --model_name ${MODEL_NAME} \
                         --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} --dist y \
                         --resume_continue y --resume ./checkpoints/cls/imagenet/${CHECKPOINTS_NAME}_latest.pth \
                         --backbone ${BACKBONE} --checkpoints_name ${CHECKPOINTS_NAME}  2>&1 | tee ${LOG_FILE}

elif [ $1 == test ]; then
  ${PYTHON} main.py --config_file ${CONFIG_FILE} --phase test --gpu 0 1 2 3 --gather n \
                    --model_name ${MODEL_NAME} --backbone ${BACKBONE} --checkpoints_name ${CHECKPOINTS_NAME} \
                    --resume ./checkpoints/cls/imagenet/${CHECKPOINTS_NAME}_latest.pth \
                    --test_dir ${DATA_DIR}/test --out_dir test  2>&1 | tee -a ${LOG_FILE}

else
  echo "Only support train & resume & test..."
fi
