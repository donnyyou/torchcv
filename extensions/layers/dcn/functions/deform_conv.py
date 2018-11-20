import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from .._ext import deform_conv


def conv_offset2d(input,
                  offset,
                  weight,
                  stride=1,
                  padding=0,
                  dilation=1,
                  deform_groups=1):

    if input is not None and input.dim() != 4:
        raise ValueError(
            "Expected 4D tensor as input, got {}D tensor instead.".format(
                input.dim()))

    f = ConvOffset2dFunction(
        _pair(stride), _pair(padding), _pair(dilation), deform_groups)
    return f(input, offset, weight)


class ConvOffset2dFunction(Function):
    def __init__(self, stride, padding, dilation, deformable_groups=1):
        super(ConvOffset2dFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

    def forward(self, input, offset, weight):
        self.save_for_backward(input, offset, weight)

        output = input.new(*self._output_size(input, weight))

        self.bufs_ = [input.new(), input.new()]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not isinstance(input.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            else:
                if not isinstance(input, torch.cuda.FloatTensor):
                    raise NotImplementedError
            deform_conv.deform_conv_forward_cuda(
                input, weight, offset, output, self.bufs_[0], self.bufs_[1],
                weight.size(3), weight.size(2), self.stride[1], self.stride[0],
                self.padding[1], self.padding[0], self.dilation[1],
                self.dilation[0], self.deformable_groups)
        return output

    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(grad_output, torch.autograd.Variable):
                if not isinstance(grad_output.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            else:
                if not isinstance(grad_output, torch.cuda.FloatTensor):
                    raise NotImplementedError
            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = input.new(*input.size()).zero_()
                grad_offset = offset.new(*offset.size()).zero_()
                deform_conv.deform_conv_backward_input_cuda(
                    input, offset, grad_output, grad_input,
                    grad_offset, weight, self.bufs_[0], weight.size(3),
                    weight.size(2), self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0], self.dilation[1],
                    self.dilation[0], self.deformable_groups)

            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                deform_conv.deform_conv_backward_parameters_cuda(
                    input, offset, grad_output,
                    grad_weight, self.bufs_[0], self.bufs_[1], weight.size(3),
                    weight.size(2), self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0], self.dilation[1],
                    self.dilation[0], self.deformable_groups, 1)

        return grad_input, grad_offset, grad_weight

    def _output_size(self, input, weight):
        channels = weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size
