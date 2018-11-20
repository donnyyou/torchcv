int deform_conv_forward(THFloatTensor *input, THFloatTensor *offset,
                        THFloatTensor *output);
int deform_conv_backward(THFloatTensor *grad_output, THFloatTensor *grad_input,
                         THFloatTensor *grad_offset);
