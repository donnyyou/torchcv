import torch
import torch.nn as nn

from model.gan.tools.image_pool import ImagePool
from model.gan.modules.subnet_selector import SubNetSelector
from model.gan.loss.gan_modules import GANLoss


class Pix2Pix(nn.Module):
    def __init__(self, configer):
        super(Pix2Pix, self).__init__()
        self.configer = configer
        # load/define networks
        self.netG = SubNetSelector.generator(net_dict=self.configer.get('network', 'generator'),
                                             use_dropout=self.configer.get('network', 'use_dropout'),
                                             norm_type=self.configer.get('network', 'norm_type'))
        self.netD = SubNetSelector.discriminator(net_dict=self.configer.get('network', 'discriminator'),
                                                 norm_type=self.configer.get('network', 'norm_type'))

        self.fake_AB_pool = ImagePool(self.configer.get('network', 'imgpool_size'))
        # define loss functions
        self.criterionGAN = GANLoss(gan_mode=self.configer.get('loss', 'params')['gan_mode'])
        self.criterionL1 = nn.L1Loss()

    def forward(self, data_dict, testing=False):
        if testing:
            out_dict = dict()
            if 'imgA' in data_dict:
                out_dict['realA'] = data_dict['imgA']
                out_dict['fakeB'] = self.netG.forward(data_dict['imgA'])

            if 'imgB' in data_dict:
                out_dict['realB'] = data_dict['imgB']

            return out_dict

        # First, G(A) should fake the discriminator
        fake_B = self.netG.forward(data_dict['imgA'])
        G_fake_AB = torch.cat((data_dict['imgA'], fake_B), 1)
        pred_fake = self.netD.forward(G_fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = self.criterionL1(fake_B, data_dict['imgB']) * self.configer.get('loss', 'loss_weights')['l1_loss']
        loss_G = loss_G_GAN + loss_G_L1

        D_fake_AB = self.fake_AB_pool.query(torch.cat((data_dict['imgA'], fake_B), 1))
        pred_fake = self.netD.forward(D_fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        D_real_AB = torch.cat((data_dict['imgA'], data_dict['imgB']), 1)
        self.pred_real = self.netD.forward(D_real_AB)
        loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return dict(loss_G=loss_G, loss_D=loss_D)
