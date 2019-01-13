// All functions assume that input and output tensors are already initialized
// and have the correct dimensions
#include <THC/THC.h>

// Forward definition of implementation functions
extern "C" {
int _ca_forward_cuda(int N, int C, int H, int W, const float *t, const float *f, float *weight, cudaStream_t);
int _ca_backward_cuda(int N, int C, int H, int W, const float *dw, const float *t, const float *f, float *dt, float *df, cudaStream_t);

int _ca_map_forward_cuda(int N, int C, int H, int W, const float *weight, const float *g, float *out, cudaStream_t);
int _ca_map_backward_cuda(int N, int C, int H, int W, const float *dout, const float *weight, const float *g, float *dw, float *dg, cudaStream_t);
}

extern THCState *state;

void get_sizes(const THCudaTensor *t, int *N, int *C, int *H, int *W){
  // Get sizes
  *N = THCudaTensor_size(state, t, 0);
  *C = THCudaTensor_size(state, t, 1);
  *H = THCudaTensor_size(state, t, 2);
  *W = THCudaTensor_size(state, t, 3);
}

extern "C" int ca_forward_cuda(const THCudaTensor *t, const THCudaTensor *f, THCudaTensor *weight) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int N, C, H, W;
  get_sizes(t, &N, &C, &H, &W);

  // Get pointers
  const float *t_data = THCudaTensor_data(state, t);
  const float *f_data = THCudaTensor_data(state, f);
  float *weight_data = THCudaTensor_data(state, weight);


  return _ca_forward_cuda(N, C, H, W, t_data, f_data, weight_data, stream);
}

extern "C" int ca_backward_cuda(const THCudaTensor *dw, const THCudaTensor *t, const THCudaTensor *f, THCudaTensor *dt, THCudaTensor *df) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int N, C, H, W;
  get_sizes(t, &N, &C, &H, &W);

  // Get pointers
  const float *dw_data = THCudaTensor_data(state, dw);
  const float *t_data = THCudaTensor_data(state, t);
  const float *f_data = THCudaTensor_data(state, f);
  float *dt_data = THCudaTensor_data(state, dt);
  float *df_data = THCudaTensor_data(state, df);


  return _ca_backward_cuda(N, C, H, W, dw_data, t_data, f_data, dt_data, df_data, stream);
}


extern "C" int ca_map_forward_cuda(const THCudaTensor *weight, const THCudaTensor *g, THCudaTensor *out) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int N, C, H, W;
  get_sizes(g, &N, &C, &H, &W);

  const float *weight_data = THCudaTensor_data(state, weight);
  const float *g_data = THCudaTensor_data(state, g);
  float *out_data = THCudaTensor_data(state, out);

  return _ca_map_forward_cuda(N, C, H, W, weight_data, g_data, out_data, stream);
}


extern "C" int ca_map_backward_cuda(const THCudaTensor *dout, const THCudaTensor *weight, const THCudaTensor *g,
                     THCudaTensor *dw,  THCudaTensor *dg) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int N, C, H, W;
  get_sizes(dout, &N, &C, &H, &W);

  const float *dout_data = THCudaTensor_data(state, dout);
  const float *weight_data = THCudaTensor_data(state, weight);
  const float *g_data = THCudaTensor_data(state, g);
  float *dw_data = THCudaTensor_data(state, dw);
  float *dg_data = THCudaTensor_data(state, dg);

  return _ca_map_backward_cuda(N, C, H, W, dout_data, weight_data, g_data, dw_data, dg_data, stream);
}

