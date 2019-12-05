#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'


CIFAR10_ROOT_DIR='/home/donny/DataSet/Cifar10/cifar-10-batches-py'
CIFAR10_SAVE_DIR='/home/donny/DataSet/Cifar10'
CIFAR100_ROOT_DIR='/home/donny/DataSet/Cifar100/cifar-100-python'
CIFAR100_SAVE_DIR='/home/donny/DataSet/Cifar100'


python cifar_cls_generator.py --root_dir $CIFAR10_ROOT_DIR \
                              --save_dir $CIFAR10_SAVE_DIR \
                              --dataset "cifar10"

python cifar_cls_generator.py --root_dir $CIFAR100_ROOT_DIR \
                              --save_dir $CIFAR100_SAVE_DIR \
                              --dataset "cifar100"
