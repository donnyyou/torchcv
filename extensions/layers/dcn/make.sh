#!/usr/bin/env bash
cd src
nvcc -c -o deform_conv_cuda_kernel.cu.o deform_conv_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++11
cd ..
CC=g++ python build.py
