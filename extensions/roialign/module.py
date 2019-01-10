import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.autograd import Variable
import os
from torch.autograd.function import once_differentiable

torch_ver = torch.__version__[:3]

from torch.utils.cpp_extension import load

print('compiling/loading roi_align')
build_path = '/tmp/bulid'
if not os.path.exists(build_path):
    os.makedirs(build_path)

roialign = load(name='roialign', sources=['extensions/roialign/src/roi_align_binding.cpp',
                                          'extensions/roialign/src/roi_align_forward_cuda.cu',
                                          'extensions/roialign/src/roi_align_backward_cuda.cu'],
                build_directory=build_path, verbose=True)


class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale, sampling_ratio):
        # ctx.save_for_backward(rois)
        ctx.rois = rois
        ctx.features_size = features.size()
        ctx.pooled_height = pooled_height
        ctx.pooled_width = pooled_width
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio

        # compute
        if features.is_cuda != rois.is_cuda:
            raise TypeError('features and rois should be on same device (CPU or GPU)')

        elif features.is_cuda and rois.is_cuda:
            output = roialign.roi_align_forward_cuda(features,
                                                     rois,
                                                     pooled_height,
                                                     pooled_width,
                                                     spatial_scale,
                                                     sampling_ratio)

        elif features.is_cuda == False and rois.is_cuda == False:
            output = roialign.roi_align_forward_cpu(features,
                                                    rois,
                                                    pooled_height,
                                                    pooled_width,
                                                    spatial_scale,
                                                    sampling_ratio)

        else:
            raise TypeError('features and rois should be on same device (CPU or GPU)')

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # rois, = ctx.saved_variables
        rois = ctx.rois
        features_size = ctx.features_size
        pooled_height = ctx.pooled_height
        pooled_width = ctx.pooled_width
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio

        # rois = ctx.rois
        if rois.is_cuda:
            grad_input = roialign.roi_align_backward_cuda(rois,
                                                          grad_output,
                                                          features_size[0],
                                                          features_size[1],
                                                          features_size[2],
                                                          features_size[3],
                                                          pooled_height,
                                                          pooled_width,
                                                          spatial_scale,
                                                          sampling_ratio)

        else:
            grad_input = roialign.roi_align_backward_cpu(rois,
                                                         grad_output,
                                                         features_size[0],
                                                         features_size[1],
                                                         features_size[2],
                                                         features_size[3],
                                                         pooled_height,
                                                         pooled_width,
                                                         spatial_scale,
                                                         sampling_ratio)

        return grad_input, None, None, None, None, None


class RoIAlign2D(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, sampling_ratio=0):
        super(RoIAlign2D, self).__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois, scale=None):
        # features is a Variable/FloatTensor of size BxCxHxW
        # rois is a (optional: list of) Variable/FloatTensor IDX,Xmin,Ymin,Xmax,Ymax (normalized to [0,1])
        rois = preprocess_rois(rois)
        if scale is None:
            output = RoIAlignFunction.apply(features,
                                            rois,
                                            self.pooled_height,
                                            self.pooled_width,
                                            self.spatial_scale,
                                            self.sampling_ratio)
        else:
            output = RoIAlignFunction.apply(features,
                                            rois,
                                            self.pooled_height,
                                            self.pooled_width,
                                            scale,
                                            self.sampling_ratio)

        return output


def preprocess_rois(rois):
    # do some verifications on what has been passed as rois
    if isinstance(rois, list):  # if list, convert to single tensor (used for multiscale)
        rois = torch.cat(tuple(rois), 0)
    if isinstance(rois, Variable):
        if rois.dim() == 3:
            if rois.size(0) == 1:
                rois = rois.squeeze(0)
            else:
                raise ("rois has wrong size")

        if rois.size(1) == 4:
            # add zeros
            zeros = Variable(torch.zeros((rois.size(0), 1)))
            if rois.is_cuda:
                zeros = zeros.cuda()

            rois = torch.cat((zeros, rois), 1).contiguous()
    return rois
