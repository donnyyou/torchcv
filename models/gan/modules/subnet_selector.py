import torch
import torch.nn as nn
from torch.nn import init
import functools

import numpy as np
import math
import torch.nn.functional as F

from collections import namedtuple


###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        print(m)
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        n = (m.in_features + m.out_features) / 2
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_4blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4,
                               gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_3blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3,
                               gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids)
    elif which_model_netG == 'lcnn':
        netG = LightCnnFeatureGenerator(gpu_ids=gpu_ids)
    elif which_model_netG == 'lcnn_conv':
        netG = LightCnnNoFCFeatureGenerator(gpu_ids=gpu_ids)
    elif which_model_netG == 'lcnn_fc1':
        netG = LightCnnFC1Generator(gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])

    print_network(netG)
    print(netG.state_dict().keys())

    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    elif which_model_netD == '2_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=2, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    elif which_model_netD == 'lcnn_feat_D':
        netD = FC_Discriminator(input_size=256, output_size=1)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def define_F(num_classes, gpu_ids=[], update_paras=False):
    netF = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
    netF = LightCnnGenerator(num_classes, gpu_ids)

    if use_gpu:
        netF.cuda(device_id=gpu_ids[0])
    netF.apply(weights_init)

    if not update_paras:
        netF.eval()
    return netF


def define_ResNeXt_F(num_classes, input_size, input_channel, gpu_ids=[]):
    netF = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
        # def __init__(self, input_size=224, in_nc=3, num_classes=1000, n_conv0=64, n_conv_stage1=256, base_width=4, rep_lists=[3,4,6,3], norm_layer=nn.BatchNorm2d, padding_type='zero', gpu_ids=[])
    netF = ResNeXtGenerator(input_size=input_size, in_nc=input_channel, num_classes=num_classes, n_conv0=64,
                            n_conv_stage1=64, rep_lists=[2, 3, 4, 2], gpu_ids=gpu_ids)

    if use_gpu:
        netF.cuda(device_id=gpu_ids[0])
    netF.apply(weights_init)
    return netF