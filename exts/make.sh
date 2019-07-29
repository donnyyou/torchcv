#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


export PATH=/opt/rh/devtoolset-3/root/usr/bin:$PATH

PYTHON=${PYTHON:-"python"}

echo "Building cocoapi..."
cd apis/cocoapi/PythonAPI
${PYTHON} setup.py install

cd -
echo "Building roi align op..."
cd ops/roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building dcn..."
cd ../dcn
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building GPU SyncBN..."
cd ../sync_bn/src/gpu
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
