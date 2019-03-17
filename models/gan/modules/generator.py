import torch
import torch.nn as nn

from models.tools.module_helper import ModuleHelper

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_type=None, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 ModuleHelper.BatchNorm2d(norm_type=norm_type)(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      ModuleHelper.BatchNorm2d(norm_type=norm_type)(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_type=norm_type, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      ModuleHelper.BatchNorm2d(norm_type=norm_type)(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_type, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_type, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_type, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       ModuleHelper.BatchNorm2d(norm_type=norm_type)(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       ModuleHelper.BatchNorm2d(norm_type=norm_type)(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_type=None, use_dropout=False):
        super(UNetGenerator, self).__init__()

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, norm_type=norm_type, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block,
                                                 norm_type=norm_type, use_dropout=use_dropout)

        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_type=norm_type)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_type=norm_type)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_type=norm_type)
        unet_block = UNetSkipConnectionBlock(output_nc, ngf, submodule=unet_block, outermost=True, norm_type=norm_type)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, submodule=None,
                 outermost=False, innermost=False, norm_type=None, use_dropout=False):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = ModuleHelper.BatchNorm2d(norm_type=norm_type)(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = ModuleHelper.BatchNorm2d(norm_type=norm_type)(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)