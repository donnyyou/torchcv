template <typename DType>
void deformable_im2col(cudaStream_t stream, const DType *data_im,
                       const DType *data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, DType *data_col);

template <typename DType>
void deformable_col2im(cudaStream_t stream, const DType *data_col,
                       const DType *data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, DType *grad_im);

template <typename DType>
void deformable_col2im_coord(cudaStream_t stream, const DType *data_col,
                             const DType *data_im, const DType *data_offset,
                             const int channels, const int height,
                             const int width, const int ksize_h,
                             const int ksize_w, const int pad_h,
                             const int pad_w, const int stride_h,
                             const int stride_w, const int dilation_h,
                             const int dilation_w, const int deformable_group,
                             DType *grad_offset);
