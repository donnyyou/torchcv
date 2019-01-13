#ifndef __CA__
#define __CA__

/*
 * Exported functions
 */
extern "C" int _ca_forward_cuda(int N, int C, int H, int W, const float *t, const float *f, float *weight, cudaStream_t stream);
extern "C" int _ca_backward_cuda(int N, int C, int H, int W, const float *dw, const float *t, const float *f, float *dt, float *df, cudaStream_t stream);
extern "C" int _ca_map_forward_cuda(int N, int C, int H, int W, const float *weight, const float *g, float *out, cudaStream_t stream);
extern "C" int _ca_map_backward_cuda(int N, int C, int H, int W, const float *dout, const float *weight, const float *g, float *dw, float *dg, cudaStream_t stream);

#endif
