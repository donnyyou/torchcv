int deform_conv_forward_cuda(THCudaTensor *input,
                             THCudaTensor *weight, /*THCudaTensor * bias, */
                             THCudaTensor *offset, THCudaTensor *output,
                             THCudaTensor *columns, THCudaTensor *ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW,
                             int deformable_group);

int deform_conv_backward_input_cuda(
    THCudaTensor *input, THCudaTensor *offset, THCudaTensor *gradOutput,
    THCudaTensor *gradInput, THCudaTensor *gradOffset, THCudaTensor *weight,
    THCudaTensor *columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW, int deformable_group);

int deform_conv_backward_parameters_cuda(
    THCudaTensor *input, THCudaTensor *offset, THCudaTensor *gradOutput,
    THCudaTensor *gradWeight, /*THCudaTensor *gradBias, */
    THCudaTensor *columns, THCudaTensor *ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW, int deformable_group,
    float scale);
