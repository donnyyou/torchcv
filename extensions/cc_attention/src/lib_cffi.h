int ca_forward_cuda(const THCudaTensor *t, const THCudaTensor *f, THCudaTensor *weight);

int ca_backward_cuda(const THCudaTensor *dw, const THCudaTensor *t, const THCudaTensor *f, THCudaTensor *dt, THCudaTensor *df);

int ca_map_forward_cuda(const THCudaTensor *weight, const THCudaTensor *g, THCudaTensor *out);
int ca_map_backward_cuda(const THCudaTensor *dout, const THCudaTensor *weight, const THCudaTensor *g,
                     THCudaTensor *dw,  THCudaTensor *dg);
