#include <torch/torch.h>
#include <vector>


at::Tensor BatchNorm_Forward_CUDA(
  const at::Tensor input_, 
  const at::Tensor mean_,
  const at::Tensor std_,
  const at::Tensor gamma_,
  const at::Tensor beta_);

std::vector<at::Tensor> BatchNorm_Backward_CUDA(
  const at::Tensor gradoutput_,
  const at::Tensor input_,
  const at::Tensor mean_, 
  const at::Tensor std_,
  const at::Tensor gamma_,
  const at::Tensor beta_, 
  bool train);

std::vector<at::Tensor> Sum_Square_Forward_CUDA(
  const at::Tensor input_);

at::Tensor Sum_Square_Backward_CUDA(
  const at::Tensor input_,
  const at::Tensor gradSum_,
  const at::Tensor gradSquare_);
