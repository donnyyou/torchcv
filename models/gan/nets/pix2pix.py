import torch
import torch.nn as nn

from models.gan.tools.image_pool import ImagePool
from models.gan.modules.subnet_selector import SubnetSelector
from models.gan.loss.gan_modules import GANLoss


class Pix2Pix(nn.Module):
    def initialize(self, opt):

        # load/define networks
        self.netG = SubnetSelector.generator(self.configer.get('network', 'generator'))
        self.netD = SubnetSelector.discriminator(self.configer.get('network', 'discriminator'))

        self.fake_AB_pool = ImagePool(opt.pool_size)
        # define loss functions
        self.criterionGAN = GANLoss(use_lsgan=self.configer.get('loss', 'use_lsgan'))
        self.criterionL1 = nn.L1Loss()

    def forward(self, data_dict):
        # First, G(A) should fake the discriminator
        fake_B = self.netG.forward(data_dict['imgA'])
        G_fake_AB = torch.cat((data_dict['imgA'], fake_B), 1)
        pred_fake = self.netD.forward(G_fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = self.criterionL1(fake_B, data_dict['imgB']) * self.opt.lambda_A
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

        return dict(loss=loss_G + loss_D)

    def forward_test(self, data_dict):
        return dict(fakeB=self.netG.forward(data_dict['imgA']))
