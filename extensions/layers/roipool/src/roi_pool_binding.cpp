#include <torch/torch.h>
#include "roi_pool_cpu.cpp"
#include "roi_pool_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &roi_pool_forward_cpu, "roi_pool_forward_cpu");
    m.def("backward_cpu", &roi_pool_backward_cpu, "roi_pool_backward_cpu");
    m.def("forward_cuda", &roi_pool_forward_cuda, "roi_pool_forward_cuda");
    m.def("backward_cuda", &roi_pool_backward_cuda, "roi_pool_backward_cuda");
}
