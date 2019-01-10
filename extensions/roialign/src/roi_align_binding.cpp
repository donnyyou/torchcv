#include <torch/torch.h>
#include "roi_align_cpu.cpp"
#include "roi_align_backward_cpu.cpp"
#include "roi_align_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward_cpu", &at::contrib::roi_align_forward_cpu, "roi_align_forward_cpu");
  m.def("roi_align_backward_cpu", &at::contrib::roi_align_backward_cpu, "roi_align_backward_cpu");
  m.def("roi_align_forward_cuda", &at::contrib::roi_align_forward_cuda, "roi_align_forward_cuda");
  m.def("roi_align_backward_cuda", &at::contrib::roi_align_backward_cuda, "roi_align_backward_cuda");
}
