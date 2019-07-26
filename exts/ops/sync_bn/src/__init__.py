import os
import torch
from torch.utils.cpp_extension import load

cwd = os.path.dirname(os.path.realpath(__file__))
cpu_path = os.path.join(cwd, 'cpu')
gpu_path = os.path.join(cwd, 'gpu')

from .cpu import syncbn_cpu as cpu
from .gpu import syncbn_gpu as gpu
# cpu = load('syncbn_cpu', [
#         os.path.join(cpu_path, 'operator.cpp'),
#         os.path.join(cpu_path, 'syncbn_cpu.cpp'),
#     ], build_directory=cpu_path, verbose=False)
#
# if torch.cuda.is_available():
#     gpu = load('syncbn_gpu', [
#             os.path.join(gpu_path, 'operator.cpp'),
#             os.path.join(gpu_path, 'syncbn_kernel.cu'),
#         ], build_directory=gpu_path, verbose=False)
