#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="python"

export PYTHONPATH="/home/donny/Projects/TorchCV":$PYTHONPATH

cd ../../../

DATA_DIR="/home/donny/DataSet/COCO_INS_2014"
MODEL_NAME="darknet_yolov3"
LOSS_TYPE="yolov3_loss"
CHECKPOINTS_NAME="yolov3_darknet_coco_det"$2
PRETRAINED_MODEL="./pretrained_models/darknet53_weights_pretrained.pth"
HYPES_FILE='hypes/det/coco/yolov3_darknet_coco_det.json'

LOG_DIR="./log/pose/coco/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi

if [[ "$1"x == "train"x ]]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase train --log_to_file n --gpu 0 1 2 3 --cudnn y \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee ${LOG_FILE}

elif [[ "$1"x == "resume"x ]]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase train --log_to_file n --gpu 0 1 2 3 --cudnn n \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --model_name ${MODEL_NAME} \
                       --resume_continue y --resume ./checkpoints/det/coco/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  2>&1 | tee -a ${LOG_FILE}

elif [[ "$1"x == "debug"x ]]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase debug --gpu 0 --log_to_file n  2>&1 | tee ${LOG_FILE}

elif [[ "$1"x == "val"x ]]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase test --log_to_file n --model_name ${MODEL_NAME} \
                       --phase test --gpu 0 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --out_dir val  2>&1 | tee -a ${LOG_FILE}
  cd metrics/det/
  ${PYTHON} -u voc_evaluator.py --hypes "../../../"${HYPES_FILE} \
                                --json_dir ../../../out/results/voc/test_dir/${CHECKPOINTS_NAME}/val/json \
                                --gt_dir ${DATA_DIR}/val/label  2>&1 | tee -a "../../"${LOG_FILE}

else
  echo "$1"x" is invalid..."
fi
