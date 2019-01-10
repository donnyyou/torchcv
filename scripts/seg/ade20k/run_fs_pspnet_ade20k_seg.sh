#!/usr/bin/env bash


cd ../../../


if [ "$1"x == "train"x ]; then
  python main.py --hypes hypes/ade20k/fs_pspnet_ade20k_seg.json
                 --phase train --gpu 0 1 2 3
                 --pretrained ../imagenet_101.pth

elif [ "$1"x == "debug"x ]; then
  python main.py --hypes hypes/ade20k/fs_pspnet_ade20k_seg.json --phase debug --gpu 0

elif [ "$1"x == "test"x ]; then
  python main.py --hypes hypes/ade20k/fs_pspnet_ade20k_seg.json --phase test --gpu 0 --resume $2
  cd val/scripts/seg
  python ade20k_evaluator.py --hypes_file ././../../hypes/ade20k/fs_pspnet_ade20k_seg.json
                             --gt_dir path-to-gt
                             --pred_dir path-to-pred
else
  echo "$1"x" is invalid..."
fi