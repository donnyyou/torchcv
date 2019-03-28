#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="python"

export PYTHONPATH="/home/donny/Projects/TorchCV":${PYTHONPATH}

cd ../../../

DATA_DIR="/home/donny/DataSet/ADE20K"

BACKBONE="deepbase_resnet101_dilated8"
MODEL_NAME="deeplabv3"
CHECKPOINTS_NAME="fs_res101_deeplabv3_ade20k_seg"$2
PRETRAINED_MODEL="./pretrained_models/3x3resnet101-imagenet.pth"

HYPES_FILE='hypes/seg/ade20k/fs_deeplabv3_ade20k_seg.json'
MAX_ITERS=150000
LOSS_TYPE="fs_auxce_loss"

LOG_DIR="./log/seg/ade20k/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi


if [[ "$1"x == "train"x ]]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --drop_last y --phase train --gathered n --loss_balance y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 --log_to_file n \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee ${LOG_FILE}

elif [[ "$1"x == "resume"x ]]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --drop_last y --phase train --gathered n --loss_balance y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 --log_to_file n\
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --resume_continue y --resume ./checkpoints/seg/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee -a ${LOG_FILE}

elif [[ "$1"x == "debug"x ]]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase debug --gpu 0 --log_to_file n  2>&1 | tee ${LOG_FILE}

elif [[ "$1"x == "val"x ]]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase test --gpu 0 1 2 3 --log_to_file n --gathered n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/seg/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --out_dir val  2>&1 | tee -a ${LOG_FILE}
  cd metrics/seg/
  ${PYTHON} -u ade20k_evaluator.py --hypes "../../"${HYPES_FILE} \
                                   --pred_dir ../../results/seg/ade20k/${CHECKPOINTS_NAME}/val/label \
                                   --gt_dir ${DATA_DIR}/val/label  2>&1 | tee -a "../../"${LOG_FILE}

elif [[ "$1"x == "test"x ]]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase test --gpu 0 1 2 3 --log_to_file n --gathered n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/seg/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/test --out_dir test  2>&1 | tee -a ${LOG_FILE}

else
  echo "$1"x" is invalid..."
fi
