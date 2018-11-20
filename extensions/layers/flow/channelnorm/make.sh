#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
TORCH=$($PYTHON -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src
echo "Compiling channelnorm kernels by nvcc..."
rm ChannelNorm_kernel.o
rm -r ../_ext

nvcc -c -o ChannelNorm_kernel.o ChannelNorm_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_30 -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

cd ../
$PYTHON build.py
