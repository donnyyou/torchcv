import torch
import math

from models.gan.modules.generator import ResNetGenerator, UNetGenerator
from models.gan.modules.discriminator import NLayerDiscriminator, FCDiscriminator

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



class SubnetSelector(object):

    @staticmethod
    def generator(net_dict):
        if net_dict['net_type'] == 'resnet':
            netG = ResNetGenerator(**net_dict['params'])
        elif net_dict['net_type'] == 'unet':
            netG = UNetGenerator(**net_dict['params'])
        else:
            raise NotImplementedError('Generator model name [{}] is not recognized'.format(net_dict['net_type']))

        return netG

    @staticmethod
    def discriminator(net_dict):
        if net_dict['net_type'] == 'nlayer':
            netD = NLayerDiscriminator(**net_dict['params'])
        elif net_dict['net_type'] == 'fc':
            netD = FCDiscriminator(**net_dict['params'])
        else:
            raise NotImplementedError('Generator model name [{}] is not recognized'.format(net_dict['net_type']))

        return netD

    @staticmethod
    def extrator(net_dict):
        pass

    @staticmethod
    def classifier(net_dict):
        pass
