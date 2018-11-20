namespace at {
namespace contrib {

Tensor roi_align_forward_cuda(
  const Tensor& input,
  const Tensor& bottom_rois,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio);

Tensor roi_align_backward_cuda(
  const Tensor& bottom_rois,
  const Tensor& grad_output, // gradient of the output of the layer
  int64_t b_size,
  int64_t channels,
  int64_t height,
  int64_t width,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio);


}
}