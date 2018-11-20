#include <THC/THC.h>

#include "deform_conv_cuda_kernel.h"

extern THCState *state;

void shape_check(THCState *state, THCudaTensor *input, THCudaTensor *offset,
                 THCudaTensor *gradOutput, THCudaTensor *weight, int kH, int kW,
                 int dH, int dW, int padH, int padW, int dilationH,
                 int dilationW, int deformable_group) {

  THArgCheck(weight->nDimension == 4, 5,
             "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
             "but got: %s",
             weight->nDimension);

  THArgCheck(THCudaTensor_isContiguous(state, weight), 5,
             "weight tensor has to be contiguous");

  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d",
             kH, kW);

  THArgCheck((weight->size[2] == kH && weight->size[3] == kW), 9,
             "kernel size should be consistent with weight, ",
             "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
             kW, weight->size[2], weight->size[3]);

  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH,
             dW);

  THArgCheck(
      dilationW > 0 && dilationH > 0, 14,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH, dilationW);

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THArgCheck(ndim == 3 || ndim == 4, 2,
             "3D or 4D input tensor expected but got: %s", ndim);

  long nInputPlane = weight->size[1];
  long inputHeight = input->size[dimh];
  long inputWidth = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  THArgCheck(nInputPlane % deformable_group == 0, 2,
             "input channels must divide deformable group size");

  if (outputWidth < 1 || outputHeight < 1)
    THError(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  THArgCheck(input->size[1] == nInputPlane, 2,
             "invalid number of input planes, expected: %d, but got: %d",
             nInputPlane, input->size[1]);

  THArgCheck((inputHeight >= kH && inputWidth >= kW), 2,
             "input image is smaller than kernel");

  THArgCheck(
      (offset->size[2] == outputHeight && offset->size[3] == outputWidth), 3,
      "invalid spatial size of offset, expected height: %d width: %d, but got height: %d width: %d", outputHeight, outputWidth,
      offset->size[2], offset->size[3]);

  THArgCheck((offset->size[1] == deformable_group * 2 * kH * kW), 3,
             "invalid number of channels of offset");

  if (gradOutput != NULL) {
    THArgCheck(gradOutput->size[dimf] == nOutputPlane, 4,
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, gradOutput->size[dimf]);

    THArgCheck((gradOutput->size[dimh] == outputHeight &&
                gradOutput->size[dimw] == outputWidth),
               4, "invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d", outputHeight, outputWidth,
               gradOutput->size[dimh], gradOutput->size[dimw]);
  }
}

int deform_conv_forward_cuda(THCudaTensor *input, THCudaTensor *weight,
                             THCudaTensor *offset, THCudaTensor *output,
                             THCudaTensor *columns, THCudaTensor *ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW,
                             int deformable_group) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, weight, offset,
                                         output, columns, ones));

  shape_check(state, input, offset, NULL, weight, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW, deformable_group);

  input = THCudaTensor_newContiguous(state, input);
  offset = THCudaTensor_newContiguous(state, offset);
  weight = THCudaTensor_newContiguous(state, weight);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],
                          input->size[2]);
    THCudaTensor_resize4d(state, offset, 1, offset->size[0], offset->size[1],
                          offset->size[2]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long inputHeight = input->size[2];
  long inputWidth = input->size[3];

  long nOutputPlane = weight->size[0];

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((offset->size[0] == batchSize), 3, "invalid batch size of offset");

  // bias = bias ? THCudaTensor_newContiguous(state, bias) : bias;

  THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight,
                        outputWidth);

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        outputHeight * outputWidth);

  if (ones->nDimension != 2 ||
      ones->size[0] * ones->size[1] < outputHeight * outputWidth) {
    THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *offset_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {

    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, offset_n, offset, 0, elt);
    THCudaTensor_select(state, output_n, output, 0, elt);

    // long m_ = nOutputPlane;
    // long n_ = outputHeight * outputWidth;
    // long k_ = 1;

    // TODO(BZ) add bias term
    // if (bias) {
    //   THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
    //                    THCudaTensor_data(state, ones), k_,
    //                    THCudaTensor_data(state, bias), k_, 0.0f,
    //                    THCudaTensor_data(state, output_n), n_);
    // } else {
    //   THCudaTensor_zero(state, output_n);
    // }

    THCudaTensor_zero(state, output_n);

    deformable_im2col(
        THCState_getCurrentStream(state), THCudaTensor_data(state, input_n),
        THCudaTensor_data(state, offset_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group, THCudaTensor_data(state, columns));

    long m = nOutputPlane;
    long n = columns->size[1];
    long k = nInputPlane * kH * kW;

    THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                     THCudaTensor_data(state, columns), n,
                     THCudaTensor_data(state, weight), k, 1.0f,
                     THCudaTensor_data(state, output_n), n);
  }

  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, offset_n);
  THCudaTensor_free(state, output_n);

  if (batch == 0) {
    THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, offset, offset->size[1], offset->size[2],
                          offset->size[3]);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, offset);
  THCudaTensor_free(state, weight);
  // if (bias) THCudaTensor_free(state, bias);

  return 1;
}

int deform_conv_backward_input_cuda(
    THCudaTensor *input, THCudaTensor *offset, THCudaTensor *gradOutput,
    THCudaTensor *gradInput, THCudaTensor *gradOffset, THCudaTensor *weight,
    THCudaTensor *columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW, int deformable_group) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, gradOutput, weight,
                                         offset, columns, gradInput));

  shape_check(state, input, offset, gradOutput, weight, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW, deformable_group);

  input = THCudaTensor_newContiguous(state, input);
  offset = THCudaTensor_newContiguous(state, offset);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  weight = THCudaTensor_newContiguous(state, weight);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],
                          input->size[2]);
    THCudaTensor_resize4d(state, offset, 1, offset->size[0], offset->size[1],
                          offset->size[2]);
    THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0],
                          gradOutput->size[1], gradOutput->size[2]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long inputHeight = input->size[2];
  long inputWidth = input->size[3];

  long nOutputPlane = weight->size[0];

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((offset->size[0] == batchSize), 3, "invalid batch size of offset");

  THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight,
                        inputWidth);

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        outputHeight * outputWidth);

  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *gradOffset_n = THCudaTensor_new(state);
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *offset_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, gradOffset_n, gradOffset, 0, elt);
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, offset_n, offset, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    long m = nInputPlane * kW * kH;
    long n = columns->size[1];
    long k = nOutputPlane;

    THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                     THCudaTensor_data(state, gradOutput_n), n,
                     THCudaTensor_data(state, weight), m, 0.0f,
                     THCudaTensor_data(state, columns), n);

    deformable_col2im_coord(
        THCState_getCurrentStream(state), THCudaTensor_data(state, columns),
        THCudaTensor_data(state, input_n), THCudaTensor_data(state, offset_n),
        nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
        dilationH, dilationW, deformable_group,
        THCudaTensor_data(state, gradOffset_n));

    deformable_col2im(
        THCState_getCurrentStream(state), THCudaTensor_data(state, columns),
        THCudaTensor_data(state, offset_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group, THCudaTensor_data(state, gradInput_n));
  }

  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, gradOffset_n);
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, offset_n);
  THCudaTensor_free(state, gradOutput_n);

  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, gradInput, nInputPlane, inputHeight,
                          inputWidth);
    THCudaTensor_resize3d(state, offset, offset->size[1], offset->size[2],
                          offset->size[3]);
    THCudaTensor_resize3d(state, gradOffset, offset->size[1], offset->size[2],
                          offset->size[3]);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, offset);
  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, weight);

  return 1;
}

int deform_conv_backward_parameters_cuda(
    THCudaTensor *input, THCudaTensor *offset, THCudaTensor *gradOutput,
    THCudaTensor *gradWeight, /*THCudaTensor *gradBias, */
    THCudaTensor *columns, THCudaTensor *ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW, int deformable_group,
    float scale) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, offset, gradOutput,
                                         gradWeight, columns));

  shape_check(state, input, offset, gradOutput, gradWeight, kH, kW, dH, dW,
              padH, padW, dilationH, dilationW, deformable_group);

  input = THCudaTensor_newContiguous(state, input);
  offset = THCudaTensor_newContiguous(state, offset);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],
                          input->size[2]);
    THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0],
                          gradOutput->size[1], gradOutput->size[2]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long inputHeight = input->size[2];
  long inputWidth = input->size[3];

  long nOutputPlane = gradWeight->size[0];

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((offset->size[0] == batchSize), 3, "invalid batch size of offset");

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        outputHeight * outputWidth);

  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *offset_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, offset_n, offset, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    deformable_im2col(
        THCState_getCurrentStream(state), THCudaTensor_data(state, input_n),
        THCudaTensor_data(state, offset_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group, THCudaTensor_data(state, columns));

    long m = nOutputPlane;
    long n = nInputPlane * kW * kH;
    long k = columns->size[1];

    THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
                     THCudaTensor_data(state, columns), k,
                     THCudaTensor_data(state, gradOutput_n), k, 1.0f,
                     THCudaTensor_data(state, gradWeight), n);
  }

  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, offset_n);
  THCudaTensor_free(state, gradOutput_n);

  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, offset);
  THCudaTensor_free(state, gradOutput);
  return 1;
}
