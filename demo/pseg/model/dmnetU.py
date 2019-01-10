import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torch.autograd import Variable

############### bottleneck struct ###########

class Bottleneck3x3(nn.Module):
    def __init__(self, inplanes, planes):
        super(Bottleneck3x3, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
        )
        self.prelu = nn.PReLU(inplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        out = self.prelu(out)

        return out


class Bottleneck5x5(nn.Module):
    def __init__(self, inplanes, planes):
        super(Bottleneck5x5, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
        )
        self.prelu = nn.PReLU(inplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        out = self.prelu(out)

        return out


class BottleneckDim3x3(nn.Module):  # down dim
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckDim3x3, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        self.prelu = nn.PReLU(outplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual = self.conv2(residual)
        out += residual
        out = self.prelu(out)

        return out


class BottleneckDim5x5(nn.Module):  # down dim
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckDim5x5, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        self.prelu = nn.PReLU(outplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual = self.conv2(residual)
        out += residual
        out = self.prelu(out)

        return out

class DMNetBottleF(nn.Module):
    def __init__(self, base_featuremap):
        super(DMNetBottleF, self).__init__()
        featuremap0_num = base_featuremap / 2
        featuremap1_num = base_featuremap
        featuremap2_num = base_featuremap * 2
        featuremap3_num = base_featuremap * 4
        featuremap4_num = base_featuremap * 8

        self.conv1 = nn.Sequential(
            nn.Conv2d(5, featuremap1_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(featuremap1_num),
            nn.PReLU(featuremap1_num),
            Bottleneck3x3(featuremap1_num, featuremap1_num / 2),
        )

        self.conv2 = nn.Sequential(
            BottleneckDim3x3(featuremap1_num, featuremap1_num / 2, featuremap2_num),
            Bottleneck3x3(featuremap2_num, featuremap2_num / 2),
        )

        self.conv3 = nn.Sequential(
            BottleneckDim3x3(featuremap2_num, featuremap2_num / 2, featuremap3_num),
            Bottleneck3x3(featuremap3_num, featuremap3_num / 2),
            Bottleneck3x3(featuremap3_num, featuremap3_num / 2),
        )

        self.conv4 = nn.Sequential(
            BottleneckDim3x3(featuremap3_num, featuremap3_num / 2, featuremap4_num),
            Bottleneck3x3(featuremap4_num, featuremap4_num / 4),
            Bottleneck3x3(featuremap4_num, featuremap4_num / 4),
        )

        self.conv5 = nn.Sequential(
            Bottleneck3x3(featuremap4_num, featuremap4_num / 4),
            Bottleneck3x3(featuremap4_num, featuremap4_num / 4),
            Bottleneck3x3(featuremap4_num, featuremap4_num / 4),
        )

        self.deconv5 = Bottleneck5x5(featuremap4_num, featuremap4_num / 4)
        self.deconv4 = BottleneckDim5x5(featuremap4_num, featuremap4_num / 4, featuremap3_num)
        self.deconv3 = BottleneckDim5x5(featuremap3_num, featuremap2_num / 2, featuremap2_num)
        self.deconv2 = BottleneckDim5x5(featuremap2_num, featuremap1_num / 2, featuremap1_num)
        self.deconv1 = Bottleneck5x5(featuremap1_num, featuremap1_num / 2)

        self.deconv0 = nn.Sequential(
            nn.Conv2d(featuremap1_num, featuremap0_num, kernel_size=1),
            nn.BatchNorm2d(featuremap0_num),
            nn.PReLU(featuremap0_num),

            nn.Conv2d(featuremap0_num, featuremap0_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(featuremap0_num),
            nn.PReLU(featuremap0_num),

            nn.Conv2d(featuremap0_num, featuremap0_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(featuremap0_num),
            nn.PReLU(featuremap0_num),

            nn.Conv2d(featuremap0_num, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x, idx1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.conv2(x)
        x, idx2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.conv3(x)
        x, idx3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.conv4(x)
        x, idx4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.conv5(x)
        # x, idx5 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # x = F.max_unpool2d(x, idx5, kernel_size=2, stride=2)
        x = self.deconv5(x)

        x = F.max_unpool2d(x, idx4, kernel_size=2, stride=2)
        x = self.deconv4(x)

        x = F.max_unpool2d(x, idx3, kernel_size=2, stride=2)
        x = self.deconv3(x)

        x = F.max_unpool2d(x, idx2, kernel_size=2, stride=2)
        x = self.deconv2(x)

        x = F.max_unpool2d(x, idx1, kernel_size=2, stride=2)
        x = self.deconv1(x)

        x = self.deconv0(x)
        return x

class DMNetBottleFSimple(nn.Module):
    def __init__(self, base_featuremap):
        super(DMNetBottleFSimple, self).__init__()
        featuremap0_num = base_featuremap // 2
        featuremap1_num = base_featuremap
        featuremap2_num = base_featuremap * 2
        featuremap3_num = base_featuremap * 4
        featuremap4_num = base_featuremap * 8

        self.conv1 = nn.Sequential(
            nn.Conv2d(5, featuremap1_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(featuremap1_num),
            nn.PReLU(featuremap1_num),
            #Bottleneck3x3(featuremap1_num, featuremap1_num / 2),
        )

        self.conv2 = nn.Sequential(
            BottleneckDim3x3(featuremap1_num, featuremap1_num // 2, featuremap2_num),
            #Bottleneck3x3(featuremap2_num, featuremap2_num / 2),
        )

        self.conv3 = nn.Sequential(
            BottleneckDim3x3(featuremap2_num, featuremap2_num // 2, featuremap3_num),
            #Bottleneck3x3(featuremap3_num, featuremap3_num / 2),
            #Bottleneck3x3(featuremap3_num, featuremap3_num / 2),
        )

        self.conv4 = nn.Sequential(
            BottleneckDim3x3(featuremap3_num, featuremap3_num // 2, featuremap4_num),
            #Bottleneck3x3(featuremap4_num, featuremap4_num / 4),
            #Bottleneck3x3(featuremap4_num, featuremap4_num / 4),
        )

        self.conv5 = nn.Sequential(
            Bottleneck3x3(featuremap4_num, featuremap4_num // 4),
            #Bottleneck3x3(featuremap4_num, featuremap4_num / 4),
            #Bottleneck3x3(featuremap4_num, featuremap4_num / 4),
        )

        self.deconv5 = Bottleneck3x3(featuremap4_num, featuremap4_num // 4)
        self.deconv4 = BottleneckDim3x3(featuremap4_num, featuremap4_num // 4, featuremap3_num)
        self.deconv3 = BottleneckDim3x3(featuremap3_num, featuremap2_num // 2, featuremap2_num)
        self.deconv2 = BottleneckDim3x3(featuremap2_num, featuremap1_num // 2, featuremap1_num)
        self.deconv1 = Bottleneck3x3(featuremap1_num, featuremap1_num // 2)

        self.deconv0 = nn.Sequential(
            nn.Conv2d(featuremap1_num, featuremap0_num, kernel_size=1),
            nn.BatchNorm2d(featuremap0_num),
            nn.PReLU(featuremap0_num),

            nn.Conv2d(featuremap0_num, featuremap0_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(featuremap0_num),
            nn.PReLU(featuremap0_num),

            # nn.Conv2d(featuremap0_num, featuremap0_num, kernel_size=3, padding=1),
            # nn.BatchNorm2d(featuremap0_num),
            # nn.PReLU(featuremap0_num),

            nn.Conv2d(featuremap0_num, 1, kernel_size=1),
        )
    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if own_state[name].size() == param.size():
                own_state[name].copy_(param)

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def forward(self, x):
        x = self.conv1(x)
        x, idx1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.conv2(x)
        x, idx2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.conv3(x)
        x, idx3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.conv4(x)
        x, idx4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        x = self.conv5(x)

        x = self.deconv5(x)

        x = F.max_unpool2d(x, idx4, kernel_size=2, stride=2)
        x = self.deconv4(x)

        x = F.max_unpool2d(x, idx3, kernel_size=2, stride=2)
        x = self.deconv3(x)

        x = F.max_unpool2d(x, idx2, kernel_size=2, stride=2)
        x = self.deconv2(x)

        x = F.max_unpool2d(x, idx1, kernel_size=2, stride=2)
        x = self.deconv1(x)

        x = self.deconv0(x)
        return x


class l_Net(torch.nn.Module):
    def __init__(self, base_featuremap, ref_featuremap, dmTpye = 'big', model_path=""): # type: small, big
        super(l_Net, self).__init__()
        if dmTpye == 'big':
            self.dmnet = DMNetBottleF(base_featuremap)
        elif dmTpye == 'small':
            self.dmnet = DMNetBottleFSimple(base_featuremap)
        if model_path:
            self.dmnet.load_state_dict(torch.load(model_path))
            for param in self.dmnet.parameters():
                param.requires_grad = False
        self.rfnet = nn.Sequential(
            nn.Conv2d(4, ref_featuremap, kernel_size=3, padding=1),
            nn.PReLU(ref_featuremap),
            nn.Conv2d(ref_featuremap, ref_featuremap, kernel_size=3, padding=1),
            nn.PReLU(ref_featuremap),
            nn.Conv2d(ref_featuremap, ref_featuremap, kernel_size=3, padding=1),
            nn. PReLU(ref_featuremap),
            nn.Conv2d(ref_featuremap, 1, kernel_size=3, padding=1)
        )

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if own_state[name].size() == param.size():
                own_state[name].copy_(param)

        missing = set(own_state.keys()) - set(state_dict.keys())
        # if len(missing) > 0:
        #     raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def forward(self, x):
        rgb = x[:, :3, :, :]

        residual = self.dmnet(x)
        residual = torch.clamp(residual, 0, 1)
        ref_in = torch.cat((rgb, residual), 1)

        out = self.rfnet(ref_in)
        out += residual
        return residual, out
        # rgb = x[:, :3, :, :]
        # trimap = x[:, 3, :, :]
        #
        # residual = self.dmnet(x)
        # residual = torch.clamp(residual, 0, 1)
        # # residual_cat = residual * 255
        #
        # idx = torch.ones(trimap.size())
        # idx[trimap.data.cpu() < 0.1] = 0
        # idx[trimap.data.cpu() > 0.9] = 0
        #
        # idx = Variable(idx, requires_grad=False).cuda(rgb.get_device())
        # residual_cat = residual.mul(idx) + trimap.mul(1 - idx)
        #
        # ref_in = torch.cat((rgb, residual_cat), 1)
        #
        # out = self.rfnet(ref_in)
        # out += residual_cat
        # return residual, out