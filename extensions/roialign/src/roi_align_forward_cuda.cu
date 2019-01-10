// Adapted from https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_align_op.cu
// (Ignacio Rocco)

#include "ATen/NativeFunctions.h"
#include <cfloat>

namespace at {
namespace contrib {

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;
const int CUDA_MAX_BLOCKS = 65535;

inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__host__ __device__ __forceinline__ float fmin(float a, float b) {
  return a > b ? b : a;
}

__host__ __device__ __forceinline__ float fmax(float a, float b) {
  return a > b ? a : b;
}

template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void roi_align_forward_kernel(
    const int outputElements,
    const T* bottom_data, // input tensor
    const T* bottom_rois, // input rois
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* top_data) // output
 {
//  CUDA_1D_KERNEL_LOOP(index, nthreads) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < outputElements;
       index += blockDim.x * gridDim.x)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = fmax(roi_end_w - roi_start_w, (T)1.);
    T roi_height = fmax(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceilf(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}


Tensor roi_align_forward_cuda(
  const Tensor& input,
  const Tensor& bottom_rois,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio)
{

  // Input is the output of the last convolutional layer in the Backbone network, so
  // it should be in the format of NCHW
  AT_CHECK(input.ndimension() == 4, "Input to RoI Align should be a NCHW Tensor");

  // ROIs is the set of region proposals to process. It is a 2D Tensor where the first
  // dim is the # of proposals, and the second dim is the n itself in the form
  // [batch_index startW startH endW endH]
  AT_CHECK(bottom_rois.ndimension() == 2, "RoI Proposals should be a 2D Tensor, (batch_sz x proposals)");
  AT_CHECK(bottom_rois.size(1) == 5, "Proposals should be of the form [batch_index startW startH endW enH]");

  auto proposals = bottom_rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  // Output Tensor is (num_rois, C, pooled_height, pooled_width)
  auto output = input.type().tensor({proposals, channels, pooled_height, pooled_width});

  AT_CHECK(input.is_contiguous(), "input must be contiguous");
  AT_CHECK(bottom_rois.is_contiguous(), "bottom_rois must be contiguous");

  // dim3 block(512);
  // dim3 grid((output.numel() + 512 - 1) / 512);
  int64_t total_threads = output.numel();
  int64_t blocks = fmin(GET_BLOCKS(total_threads),CUDA_MAX_BLOCKS);
  
  roi_align_forward_kernel<<<blocks, CUDA_NUM_THREADS, 0, globalContext().getCurrentCUDAStream()>>>(
    output.numel(), 
    input.data<float>(), 
    bottom_rois.data<float>(), 
    static_cast<float>(spatial_scale), 
    channels,
    height, 
    width, 
    pooled_height, 
    pooled_width, 
    sampling_ratio,
    output.data<float>());
  AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_align_forward_kernel failed");

  return output;
}


} // at::contrib
} // at