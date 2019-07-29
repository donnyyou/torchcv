import torch.nn as nn

from model.gan.tools.image_pool import ImagePool
from model.gan.modules.subnet_selector import SubNetSelector
from model.gan.loss.gan_modules import GANLoss


class CycleGAN(nn.Module):

    def __init__(self, configer):
        super(CycleGAN, self).__init__()
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.configer = configer
        self.netG_A = SubNetSelector.generator(net_dict=self.configer.get('network', 'generatorA'),
                                               use_dropout=self.configer.get('network', 'use_dropout'),
                                               norm_type=self.configer.get('network', 'norm_type'))
        self.netG_B = SubNetSelector.generator(net_dict=self.configer.get('network', 'generatorB'),
                                               use_dropout=self.configer.get('network', 'use_dropout'),
                                               norm_type=self.configer.get('network', 'norm_type'))

        self.netD_A = SubNetSelector.discriminator(net_dict=self.configer.get('network', 'discriminatorA'),
                                                   norm_type=self.configer.get('network', 'norm_type'))
        self.netD_B = SubNetSelector.discriminator(net_dict=self.configer.get('network', 'discriminatorB'),
                                                   norm_type=self.configer.get('network', 'norm_type'))

        self.fake_A_pool = ImagePool(self.configer.get('network', 'imgpool_size'))
        self.fake_B_pool = ImagePool(self.configer.get('network', 'imgpool_size'))
        # define loss functions
        self.criterionGAN = GANLoss(gan_mode=self.configer.get('loss', 'params')['gan_mode'])
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

    def forward(self, data_dict, testing=False):
        if testing:
            out_dict = dict()
            if 'imgA' in data_dict:
                fake_B = self.netG_A.forward(data_dict['imgA'])
                rec_A = self.netG_B.forward(fake_B)
                out_dict['fakeB'] = fake_B
                out_dict['recA'] = rec_A
                out_dict['realA'] = data_dict['imgA']

            if 'imgB' in data_dict:
                fake_A = self.netG_B.forward(data_dict['imgB'])
                rec_B = self.netG_A.forward(fake_A)
                out_dict['fakeA'] = fake_A
                out_dict['recB'] = rec_B
                out_dict['realB'] = data_dict['imgB']

            return out_dict

        cycleA_loss_weight = self.configer.get('loss', 'loss_weights')['cycleA_loss']
        cycleB_loss_weight = self.configer.get('loss', 'loss_weights')['cycleB_loss']
        idtA_loss_weight = self.configer.get('loss', 'loss_weights')['idtA_loss']
        idtB_loss_weight = self.configer.get('loss', 'loss_weights')['idtB_loss']
        # Identity loss
        # G_A should be identity if real_B is fed.
        idt_B = self.netG_A.forward(data_dict['imgB'])
        loss_idt_A = self.criterionIdt(idt_B, data_dict['imgB']) * idtB_loss_weight
        # G_B should be identity if real_A is fed.
        idt_A = self.netG_B.forward(data_dict['imgA'])
        loss_idt_B = self.criterionIdt(idt_A, data_dict['imgA']) * idtA_loss_weight

        # GAN loss
        # D_A(G_A(A))
        fake_B = self.netG_A.forward(data_dict['imgA'])
        pred_fake_B = self.netD_B.forward(fake_B)
        loss_G_A = self.criterionGAN(pred_fake_B, True)

        fake_A = self.netG_B.forward(data_dict['imgB'])
        pred_fake_A = self.netD_A.forward(fake_A)
        loss_G_B = self.criterionGAN(pred_fake_A, True)
        # Forward cycle loss
        rec_A = self.netG_B.forward(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, data_dict['imgA']) * cycleA_loss_weight
        # Backward cycle loss
        rec_B = self.netG_A.forward(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, data_dict['imgB']) * cycleB_loss_weight
        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        D_fake_A = self.fake_A_pool.query(fake_A.clone())
        D_real_A = self.netD_A.forward(data_dict['imgA'])
        loss_D_real_A = self.criterionGAN(D_real_A, True)
        # Fake
        D_fake_A = self.netD_A.forward(D_fake_A.detach())
        loss_D_fake_A = self.criterionGAN(D_fake_A, False)
        # Combined loss
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

        D_fake_B = self.fake_B_pool.query(fake_B.clone())
        D_real_B = self.netD_B.forward(data_dict['imgB'])
        loss_D_real_B = self.criterionGAN(D_real_B, True)
        # Fake
        D_fake_B = self.netD_B.forward(D_fake_B.detach())
        loss_D_fake_B = self.criterionGAN(D_fake_B, False)
        # Combined loss
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

        return dict(loss_G=loss_G, loss_D=loss_D_A + loss_D_B)
