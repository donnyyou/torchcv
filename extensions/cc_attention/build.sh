#!/bin/bash

# Configuration
CUDA_GENCODE="\
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_50,code=sm_50"


cd src
/usr/local/cuda-8.0/bin/nvcc -I/usr/local/cuda/include --expt-extended-lambda -O3 -c -o ca.o ca.cu -x cu -Xcompiler -fPIC -std=c++11 ${CUDA_GENCODE}
cd ..
