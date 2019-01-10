#include <ATen/ATen.h>
#include <cfloat>

using std::vector;
using std::max;
using std::min;

/* -----------------------------begin of the forward---------------------------------  */
template<typename T>
void roi_pool_forward(const T *input, const T *rois, vector<int64_t> in_size, vector<int64_t> rois_size, T scale,
                      T *output, int *memory) {
    int rois_num = rois_size[0], rois_col = rois_size[1], pool_h = rois_size[2], pool_w = rois_size[3];
    int channels = in_size[1], height = in_size[2], width = in_size[3];
    int chw = channels * height * width, chw_p = channels * pool_h * pool_w;
    int *memory_data;
    for (int n = 0; n < rois_num; ++n) {
        int roi_batch_id = rois[0];
        int roi_start_w = round(rois[1] * scale);
        int roi_start_h = round(rois[2] * scale);
        int roi_end_w = round(rois[3] * scale);
        int roi_end_h = round(rois[4] * scale);
        // Force malformed ROIs to be 1x1
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);

        const T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pool_h);
        const T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pool_w);

        const T *input_data = input + roi_batch_id * chw;
        if (memory)
            memory_data = memory + n * chw_p;

        for (int c = 0; c < channels; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    // Compute pooling region for this output unit:
                    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
                    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
                    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
                    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

                    // Add roi offsets and clip to input boundaries
                    hstart = min(max(hstart + roi_start_h, 0), height);
                    hend = min(max(hend + roi_start_h, 0), height);
                    wstart = min(max(wstart + roi_start_w, 0), width);
                    wend = min(max(wend + roi_start_w, 0), width);

                    const int pool_index = ph * pool_w + pw;
                    // Define an empty pooling region to be zero
                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    output[pool_index] = is_empty ? 0 : -FLT_MAX;
                    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
                    if (memory)
                        memory_data[pool_index] = -1;

                    for (int hi = hstart; hi < hend; ++hi) {
                        for (int wi = wstart; wi < wend; ++wi) {
                            const int index = hi * width + wi;
                            if (input_data[index] > output[pool_index]) {
                                output[pool_index] = input_data[index];
                                if (memory)
                                    memory_data[pool_index] = index;
                            }
                        }
                    }
                }
            }
            // Increment all data pointers by one channel
            input_data += height * width;
            output += pool_h * pool_w;
            if (memory) memory_data += pool_h * pool_w;
        }
        rois += rois_col;
    }
}

at::Tensor roi_pool_forward_cpu(const at::Tensor &input, const at::Tensor &rois, int64_t pool_h, int64_t pool_w,
                                double scale, at::Tensor &memory) {
    AT_CHECK(input.ndimension() == 4, "Feature should be BxCxHxW forms");
    AT_CHECK(input.is_contiguous(), "Feature should be contiguous");
    AT_CHECK(rois.ndimension() == 2, "ROI Proposals should be Kx5 forms");
    AT_CHECK(rois.size(1) == 5, "ROI proposals should be Kx5 forms");
    AT_CHECK(rois.is_contiguous(), "ROI proposals should be contiguous.");

    const vector<int64_t> rois_size = {rois.size(0), rois.size(1), pool_h, pool_w};
    const vector<int64_t> input_size = {input.size(0), input.size(1), input.size(2), input.size(3)};

    auto output = input.type().tensor({rois_size[0], input_size[1], pool_h, pool_w});
    if (memory.data<int>())
        memory.zero_();

    roi_pool_forward(input.data<float>(), rois.data<float>(), input_size, rois_size, static_cast<float>(scale),
                     output.data<float>(), memory.data<int>());
    return output;
}
/* -----------------------------end of the forward---------------------------------  */

/* -----------------------------begin of the backward---------------------------------  */
template<typename T>
void roi_pool_backward(const int total, const T *grad_out, const T *rois, const int channels, const int h, const int w,
                       const int pool_h, const int pool_w, T *grad_in, const int *memory) {
    for (int idx = 0; idx < total; ++idx) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_w / pool_h) % channels;
        int n = idx / pool_w / pool_h / channels;

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
            offset_grad_in[argmax] += offset_grad_out[ph * pool_w + pw];
    }
}


at::Tensor roi_pool_backward_cpu(const at::Tensor &rois, const at::Tensor &grad_out, int64_t b_size, int64_t channel,
                                 int64_t h, int64_t w, int64_t pool_h, int64_t pool_w, at::Tensor &memory) {
    AT_CHECK(grad_out.ndimension() == 4, "Feature should be BxCxHxW forms");
    AT_CHECK(grad_out.is_contiguous(), "Feature should be contiguous");
    AT_CHECK(rois.ndimension() == 2, "ROI Proposals should be Kx5 forms");
    AT_CHECK(rois.size(1) == 5 && rois.is_contiguous(), "ROI proposals should be Kx5 forms and contiguous");
    AT_CHECK(memory.is_contiguous(), "Memory should be contiguous.");


    auto grad_in = grad_out.type().tensor({b_size, channel, h, w});
    grad_in.zero_();

    roi_pool_backward(grad_out.numel(), grad_out.data<float>(), rois.data<float>(), channel, h, w, pool_h, pool_w,
                      grad_in.data<float>(), memory.data<int>());
    return grad_in;
}