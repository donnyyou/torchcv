#include <ATen/ATen.h>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>


template<typename T>
__device__ __forceinline__

T gpu_atomic_add(const T val, T *address) {
    return atomicAdd(address, val);
}

/* ------------------------------begin of the forward--------------------------- */
template<typename T>
__global__ void
roi_pool_forward_kernel(const int total, const T *input, const T *rois, const T scale, const int channels, const int h,
                        const int w, const int pool_h, const int pool_w, T *output, int *memory) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        // int pw = idx % pool_w;
        // int ph = (idx / pool_w) % pool_h;
        // int c = (idx / pool_h / pool_w) % channels;
        // int n = idx / pool_h / pool_w / channels;

        int n = idx;
        int pw = n % pool_w;
        n /= pool_w;
        int ph = n % pool_h;
        n /= pool_h;
        int c = n % channels;
        n /= channels;

        const T *offset_rois = rois + n * 5;
        int roi_batch_idx = offset_rois[0];

        // using rounding
        int roi_start_w = round(offset_rois[1] * scale);
        int roi_start_h = round(offset_rois[2] * scale);
        int roi_end_w = round(offset_rois[3] * scale);
        int roi_end_h = round(offset_rois[4] * scale);

        // Force malformed ROIs to be 1x1
        int roi_w = max(roi_end_w - roi_start_w + 1, 1);
        int roi_h = max(roi_end_h - roi_start_h + 1, 1);
        T bin_size_h = static_cast<T>(roi_h) / static_cast<T>(pool_h);
        T bin_size_w = static_cast<T>(roi_w) / static_cast<T>(pool_w);

        int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), h);
        hend = min(max(hend + roi_start_h, 0), h);
        wstart = min(max(wstart + roi_start_w, 0), w);
        wend = min(max(wend + roi_start_w, 0), w);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        T maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;
        const T *offset_input = input + (roi_batch_idx * channels + c) * h * w;
        for (int hi = hstart; hi < hend; ++hi) {
            for (int wi = wstart; wi < wend; ++wi) {
                int ind = hi * w + wi;
                if (offset_input[ind] > maxval) {
                    maxval = offset_input[ind];
                    maxidx = ind;
                }
            }
        }
        output[idx] = maxval;
        if (memory) {
            memory[idx] = maxidx;
        }
    }
}

// TODO: there may be a bug
at::Tensor roi_pool_forward_cuda(const at::Tensor &input, const at::Tensor &rois, int64_t pool_h, int64_t pool_w,
                                 double scale, at::Tensor &memory) {
    AT_CHECK(input.ndimension() == 4, "Input features should be BxCxHxW");
    AT_CHECK(rois.ndimension() == 2 && rois.size(1) == 5, "ROIs should be Kx5 forms");

    auto rois_num = rois.size(0);
    auto channel = input.size(1), h = input.size(2), w = input.size(3);

    auto output = input.type().tensor({rois_num, channel, pool_h, pool_w});

    int64_t total = output.numel();
    const int threads = 1024;
    const int64_t blocks = (total + threads - 1) / threads > 65535 ? 65535 : (total + threads - 1) / threads;

    roi_pool_forward_kernel << < blocks, threads >> > (output.numel(), input.data<float>(), rois.data<float>(),
            static_cast<float>(scale), channel, h, w, pool_h, pool_w, output.data<float>(), memory.data<int>());

    AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_align_forward_kernel failed");
    return output;
}
/* ------------------------------end of the forward--------------------------- */

/* ------------------------------begin of the backward--------------------------- */
template<typename T>
__global__ void roi_pool_backward_kernel(const int total, const T *grad_out, const T *rois, const int channels,
                                         const int h, const int w, const int pool_h, const int pool_w, T *grad_in,
                                         const int *memory) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        // (n, c, ph, pw) is an element in the pooled output
        // int pw = idx % pool_w;
        // int ph = (idx / pool_w) % pool_h;
        // int c = (idx / pool_w / pool_h) % channels;
        // int n = idx / pool_w / pool_h / channels;

        int n = idx;
        int pw = n % pool_w;
        n /= pool_w;
        int ph = n % pool_h;
        n /= pool_h;
        int c = n % channels;
        n /= channels;

        const T *offset_rois = rois + n * 5;
        int roi_batch_idx = offset_rois[0];
        // offset of index

        int grad_in_offset = (roi_batch_idx * channels + c) * h * w;
        int grad_out_offset = (n * channels + c) * pool_h * pool_w;

        const T *offset_grad_out = grad_out + grad_out_offset;
        T *offset_grad_in = grad_in + grad_in_offset;
        const int *offset_memory = memory + grad_out_offset;

        int argmax = offset_memory[ph * pool_w + pw];
        if (argmax != -1)
            gpu_atomic_add(static_cast<T>(offset_grad_out[ph * pool_w + pw]), offset_grad_in + argmax);
    }
} // RoIPoolBackward

at::Tensor roi_pool_backward_cuda(const at::Tensor &rois, const at::Tensor &grad_out, int64_t b_size, int64_t channel,
                                  int64_t h, int64_t w, int64_t pool_h, int64_t pool_w, const at::Tensor &memory) {
    AT_CHECK(rois.ndimension() == 2 && rois.size(1) == 5, "ROIs should be Kx5 forms");
    AT_CHECK(rois.is_contiguous(), "ROIs should be contiguous");

    auto grad_in = rois.type().tensor({b_size, channel, h, w});
    grad_in.zero_();

    int64_t total = grad_out.numel();
    const int threads = 1024;
    const int64_t blocks = (total + threads - 1) / threads > 65535 ? 65535 : (total + threads - 1) / threads;

    roi_pool_backward_kernel << < blocks, threads, 0, at::globalContext().getCurrentCUDAStream() >> > (total,
            grad_out.data<float>(), rois.data<float>(), channel, h, w, pool_h, pool_w, grad_in.data<float>(),
            memory.data<int>());

    AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_align_forward_kernel failed");
    return grad_in;
}