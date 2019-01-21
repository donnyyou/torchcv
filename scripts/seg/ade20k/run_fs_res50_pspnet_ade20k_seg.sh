#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="python"

export PYTHONPATH="/home/donny/Projects/PyTorchCV":${PYTHONPATH}

cd ../../../

DATA_DIR="/home/donny/DataSet/ADE20K"

BACKBONE="deepbase_resnet50_dilated8"
MODEL_NAME="deeplabv3"
CHECKPOINTS_NAME="fs_res50_deeplabv3_ade20k_seg"$2
PRETRAINED_MODEL="./pretrained_model/resnet50-imagenet.pth"

HYPES_FILE='hypes/seg/ade20k/fs_deeplabv3_ade20k_seg.json'
MAX_ITERS=40000
LOSS_TYPE="fs_auxce_loss"

LOG_DIR="./log/seg/ade20k/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [ ! -d ${LOG_DIR} ]; then
    echo ${LOG_DIR}" not exists!!!"
    make -p ${LOG_DIR}
fi


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --drop_last y --phase train --gathered n --loss_balance y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 --log_to_file n \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} > ${LOG_FILE} 2>&1

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --drop_last y --phase train --gathered n --loss_balance y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 --log_to_file n\
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --resume_continue y --resume ./checkpoints/seg/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  >> ${LOG_FILE} 2>&1

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE}--phase debug --gpu 0 --log_to_file n  > ${LOG_FILE} 2>&1

elif [ "$1"x == "val"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase test --gpu 0  --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/seg/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --out_dir val >> ${LOG_FILE} 2>&1
  cd metrics/seg/
  ${PYTHON} -u ade20k_evaluator.py --hypes "../../"${HYPES_FILE} \
                                   --pred_dir ../../results/cityscapes/test_dir/${CHECKPOINTS_NAME}/val/label \
                                   --gt_dir ${DATA_DIR}/val/label  >> "../../"${LOG_FILE} 2>&1

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase test --gpu 0  --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/seg/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/test --out_dir test >> ${LOG_FILE} 2>&1

else
  echo "$1"x" is invalid..."
fi
