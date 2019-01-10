#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


PYTHON=${PYTHON:-"python"}

echo "Building cocoapi..."
cd apis/cocoapi/PythonAPI
${PYTHON} setup.py install

cd -
echo "Building roi align op..."
cd ./roi_align
if [ -d "build" ]; then
    rm -r build
fi
${PYTHON} setup.py build_ext --inplace
rm -r build

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
${PYTHON} setup.py build_ext --inplace
rm -r build

echo "Building nms op..."
cd ../nms/src
make clean
make PYTHON=${PYTHON}
rm -r build