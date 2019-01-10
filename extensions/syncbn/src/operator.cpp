#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batchnorm_forward", &BatchNorm_Forward_CUDA, "BatchNorm forward (CUDA)");
  m.def("batchnorm_backward", &BatchNorm_Backward_CUDA, "BatchNorm backward (CUDA)");
  m.def("sumsquare_forward", &Sum_Square_Forward_CUDA, "SumSqu forward (CUDA)");
  m.def("sumsquare_backward", &Sum_Square_Backward_CUDA, "SumSqu backward (CUDA)");
}
