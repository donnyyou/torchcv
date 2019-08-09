#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="python"

export PYTHONPATH="/home/donny/Projects/torchcv":${PYTHONPATH}

cd ../../../

DATA_DIR="/home/donny/DataSet/Cityscapes"

BACKBONE="deepbase_resnet50_dilated8"
MODEL_NAME="deeplabv3"
CHECKPOINTS_NAME="fs_deeplabv3_cityscapes_seg"$2
PRETRAINED_MODEL="./pretrained_models/3x3resnet50-imagenet.pth"

CONFIG_FILE='configs/seg/cityscapes/fs_deeplabv3_cityscapes_seg.conf'
MAX_ITERS=40000
LOSS_TYPE="dsnohemce_loss"

LOG_DIR="./log/seg/cityscapes/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi

NGPUS=4

if [[ "$1"x == "train"x ]]; then
  ${PYTHON} -m torch.distributed.launch --nproc_per_node=${NGPUS}  main.py --config_file ${CONFIG_FILE} --phase train --train_batch_size 1 --val_batch_size 1 \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --drop_last y --syncbn y --distributed y \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} 2>&1 | tee ${LOG_FILE} 

elif [[ "$1"x == "resume"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --drop_last y --phase train --gather n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu None \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --resume_continue y --resume ./checkpoints/seg/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee -a ${LOG_FILE}

elif [[ "$1"x == "val"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --phase test --gpu 0 1 2 3 --gather n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/seg/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --out_dir val  2>&1 | tee -a ${LOG_FILE}
  cd metric/seg/
  ${PYTHON} -u cityscapes_evaluator.py --pred_dir ../../results/seg/cityscapes/${CHECKPOINTS_NAME}/val/label \
                                       --gt_dir ${DATA_DIR}/val/label  2>&1 | tee -a "../../"${LOG_FILE}

elif [[ "$1"x == "test"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --phase test --gpu 0 1 2 3 --gather n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/seg/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/test --out_dir test  2>&1 | tee -a ${LOG_FILE}

else
  echo "$1"x" is invalid..."
fi
