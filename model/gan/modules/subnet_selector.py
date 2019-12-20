import torch.nn.init as init

from model.gan.modules.generator import ResNetGenerator, UNetGenerator
from model.gan.modules.discriminator import NLayerDiscriminator, FCDiscriminator, PixelDiscriminator
from lib.tools.util.logger import Logger as Log


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    Log.info('initialize network with {}'.format(init_type))
    net.apply(init_func)  # apply the initialization function <init_func>


class SubNetSelector(object):

    @staticmethod
    def generator(net_dict, norm_type=None, use_dropout=False):
        if net_dict['net_type'] == 'resnet_9blocks':
            netG = ResNetGenerator(net_dict['in_c'], net_dict['out_c'], net_dict['num_f'],
                                   norm_type=norm_type, use_dropout=use_dropout, n_blocks=9)
        elif net_dict['net_type'] == 'resnet_6blocks':
            netG = ResNetGenerator(net_dict['in_c'], net_dict['out_c'], net_dict['num_f'],
                                   norm_type=norm_type, use_dropout=use_dropout, n_blocks=6)
        elif net_dict['net_type'] == 'unet_128':
            netG = UNetGenerator(net_dict['in_c'], net_dict['out_c'], 7, net_dict['num_f'],
                                 norm_type=norm_type, use_dropout=use_dropout)
        elif net_dict['net_type'] == 'unet_256':
            netG = UNetGenerator(net_dict['in_c'], net_dict['out_c'], 8, net_dict['num_f'],
                                 norm_type=norm_type, use_dropout=use_dropout)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % net_dict['net_type'])

        init_weights(netG, init_type=net_dict['init_type'], init_gain=net_dict['init_gain'])
        return netG

    @staticmethod
    def discriminator(net_dict, norm_type=None):
        if net_dict['net_type'] == 'fc':
            netD = FCDiscriminator(net_dict['in_c'], net_dict['out_c'], net_dict['hidden_c'])
        elif net_dict['net_type'] == 'n_layers':  # more options
            netD = NLayerDiscriminator(net_dict['in_c'], net_dict['num_f'], net_dict['n_layers'], norm_type=norm_type)
        elif net_dict['net_type'] == 'pixel':     # classify if each pixel is real or fake
            netD = PixelDiscriminator(net_dict['in_c'], net_dict['ndf'], norm_type=norm_type)
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % net_dict['net_type'])

        init_weights(netD, init_type=net_dict['init_type'], init_gain=net_dict['init_gain'])
        return netD

    @staticmethod
    def extrator(net_dict):
        pass

    @staticmethod
    def classifier(net_dict):
        pass
