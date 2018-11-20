#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
TORCH=$($PYTHON -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src
echo "Compiling resample2d kernels by nvcc..."
rm Resample2d_kernel.o
rm -r ../_ext

nvcc -c -o Resample2d_kernel.o Resample2d_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_30 -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

cd ../
$PYTHON build.py
