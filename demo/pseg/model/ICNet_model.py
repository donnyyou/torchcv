import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
import math

def make_grid(tensor, nrow=8, padding=2):
    """
    Given a 4D mini-batch Tensor of shape (B x C x H x W),
    or a list of images all of the same size,
    makes a grid of images
    """
    tensorlist = None
    if isinstance(tensor, list):
        tensorlist = tensor
        numImages = len(tensorlist)
        size = torch.Size(torch.Size([long(numImages)]) + tensorlist[0].size())
        tensor = tensorlist[0].new(size)
        for i in range(numImages):
            tensor[i].copy_(tensorlist[i])
    if tensor.dim() == 2: # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3: # single image
        if tensor.size(0) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1: # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)
    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(nmaps / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps, width * xmaps).fill_(tensor.max())
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y*height+1+padding//2,height-padding)\
                .narrow(2, x*width+1+padding//2, width-padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding)
    ndarr = grid.byte().transpose(0,2).transpose(0,1).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


class init_s2(nn.Module):
    def __init__(self, inplanes, planes, outplanes):
        super(init_s2, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(planes, outplanes, kernel_size=3, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.convs(x)
        return out

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, outplanes, stride=1, pad = 1, dilation = 1):
        super(Bottleneck, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(planes, planes, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )
        self.inplanes  = inplanes
        self.outplanes = outplanes
        if inplanes != outplanes:
            self.convm = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outplanes),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.convs(x)
        if self.inplanes != self.outplanes:
            residual = self.convm(residual)

        out += residual
        out = self.relu(out)
        return out

class psp_pooling(nn.Module):
    def __init__(self, scale, stride, output_size):
        super(psp_pooling, self).__init__()
        self.scale = scale
        self.stride = stride
        self.output_size = output_size
        #self.res = [None]*len(self.scale)


    def forward(self, x):
        '''out = x
        for i in xrange(len(self.scale)):
            self.res[i] = F.avg_pool2d(x, self.scale[i], self.stride[i])
            self.res[i] = F.upsample_bilinear(self.res[i], size=self.output_size)
            out += self.res[i]'''
        res1 = F.avg_pool2d(x, self.scale[0], self.stride[0])
        res1_up = F.upsample_bilinear(res1, size=self.output_size)
        res2 = F.avg_pool2d(x, self.scale[1], self.stride[1])
        res2_up = F.upsample_bilinear(res2, size=self.output_size)
        res3 = F.avg_pool2d(x, self.scale[2], self.stride[2])
        res3_up = F.upsample_bilinear(res3, size=self.output_size)
        res4 = F.avg_pool2d(x, self.scale[3], self.stride[3])
        res4_up = F.upsample_bilinear(res4, size=self.output_size)
        out = res1_up + res2_up + res3_up + res4_up + x

        return out


class CFFConv2d(nn.Module): # cascade feature fusion
    def __init__(self, f1planes, f2planes, outplane):
        super(CFFConv2d, self).__init__()
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.classifier_conv = nn.Conv2d(f1planes, 2, kernel_size=1)
        self.conv_dil = nn.Sequential(
            nn.Conv2d(f1planes, outplane, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(outplane),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(f2planes, outplane, kernel_size=1),
            nn.BatchNorm2d(outplane),
        )

    def forward(self, x_f1, x_f2):
        x_f1_up = self.upsample2x(x_f1)
        out1 = self.classifier_conv(x_f1_up)
        x_f1_conv = self.conv_dil(x_f1_up)
        x_f2_conv = self.conv2(x_f2)

        out2 = F.relu(x_f1_conv + x_f2_conv)

        return out1, out2


class ICNet(nn.Module):
    def __init__(self, baseNum = 64):
        super(ICNet, self).__init__()

        #self.init_downDim = 32
        #self.init_Dim = 64
        #self.bottle1_downDim = 32
        #self.bottle1_Dim = 128
        #self.bottle2_downDim = 64
        #self.bottle2_Dim = 256
        self.psp_scale = [ (20,15), (12,9), (8, 6), (5, 5) ]
        self.psp_stride = [ (20, 15), (8, 6), (6, 4), (3, 2) ]
        self.psp_oriSize = (20, 15)

        # shared weights layers
        self.SWC_Convs = nn.Sequential(
            init_s2(3, baseNum // 2, baseNum),
            nn.MaxPool2d(3, 2, padding=1),
            # conv1_*
            Bottleneck(baseNum, baseNum // 2, baseNum * 2),
            Bottleneck(baseNum * 2, baseNum // 2, baseNum * 2),
            Bottleneck(baseNum * 2, baseNum // 2, baseNum * 2),
            Bottleneck(baseNum * 2, baseNum, baseNum * 4, stride=2),
        )

        # f1 dilated convs
        self.F1_Dilated = nn.Sequential(
            Bottleneck(baseNum * 4, baseNum, baseNum * 4),
            Bottleneck(baseNum * 4, baseNum, baseNum * 4),
            Bottleneck(baseNum * 4, baseNum, baseNum * 4),
            Bottleneck(baseNum * 4, baseNum * 2, baseNum * 8),

            Bottleneck(baseNum * 8, baseNum * 2, baseNum * 8, stride=1, pad=2, dilation=2),
            Bottleneck(baseNum * 8, baseNum * 2, baseNum * 8, stride=1, pad=2, dilation=2),
            Bottleneck(baseNum * 8, baseNum * 2, baseNum * 8, stride=1, pad=2, dilation=2),
            Bottleneck(baseNum * 8, baseNum * 2, baseNum * 8, stride=1, pad=2, dilation=2),
            Bottleneck(baseNum * 8, baseNum * 2, baseNum * 8, stride=1, pad=2, dilation=2),
            Bottleneck(baseNum * 8, baseNum * 4, baseNum * 16, stride=1, pad=4, dilation=4),

            Bottleneck(baseNum * 16, baseNum * 4, baseNum * 16, stride=1, pad=4, dilation=4),
            Bottleneck(baseNum * 16, baseNum * 4, baseNum * 16, stride=1, pad=4, dilation=4),


            #conv5-3
            psp_pooling(self.psp_scale, self.psp_stride, self.psp_oriSize),
            #
            nn.Conv2d(baseNum * 16, baseNum * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(baseNum * 4),
            nn.ReLU()
          )

        # f3 convs
        self.F3_Convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # CFF layers
        self.cff_F1 = CFFConv2d(baseNum * 4, baseNum * 4, baseNum * 2)
        self.cff_F2 = CFFConv2d(baseNum * 2, 64, baseNum * 2)

        # upsample
        self.Upsmple_Convs = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(baseNum * 2, 2, kernel_size=1)
        )

        self.Upsampling2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x1, x2):
        f2 = self.SWC_Convs(x1)
        x0 = F.avg_pool2d(f2, 3, 2, padding = 1)
        f1 = self.F1_Dilated(x0)
        out1, f_12 = self.cff_F1(f1, f2)

        f3 = self.F3_Convs(x2)
        out2, f_23 = self.cff_F2(f_12, f3)

        out3 = self.Upsmple_Convs(f_23)
        out4 = self.Upsampling2(out3)

        #return out1, out2, out3, out4
        return f_23, out4

class TrimapNet(nn.Module):
    def __init__(self, in_dim = 128, n_gpu=1):
        super(TrimapNet, self).__init__()
        self.ngpu = n_gpu

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    # def load_state_dict(self, state_dict):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name not in own_state:
    #             continue
    #         if isinstance(param, torch.nn.Parameter):
    #             # backwards compatibility for serialized parameters
    #             param = param.data
    #         own_state[name].copy_(param)
    #
    #     missing = set(own_state.keys()) - set(state_dict.keys())
    #     if len(missing) > 0:
    #         raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def forward(self, input):

        out = self.cls_conv(input)

        return out

if __name__ == '__main__':
    icnet = ICNet()

    x1 = Variable(torch.rand((4, 3, 320, 240)))
    x2 = Variable(torch.rand((4, 3, 640, 480)))
    out1, out2, out3, out4 = icnet(x1, x2)

