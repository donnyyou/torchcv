import torch
import torch.nn as nn


class CMD_K_Loss(nn.Module):
    def __init__(self, K=1, p=2):
        super(CMD_K_Loss, self).__init__()
        assert(K>0)
        assert(p>0)
        self.K=K
        self.p=p

    def forward(self, inputA, inputB):
        meanA = inputA.mean(0, keepdim=True)
        meanB = inputB.mean(0, keepdim=True)

        assert(meanA.size()==meanB.size())

        if self.K==1:
            loss = meanA.sub(meanB).norm(self.p)
        else:
            meanA = meanA.expand_as(inputA)
            meanB = meanB.expand_as(inputB)

            distA = inputA.sub(meanA)
            distB = inputB.sub(meanB)

            CM_A = distA.pow(self.K).mean(0)
            CM_B = distB.pow(self.K).mean(0)

            loss = CM_A.sub(CM_B).norm(self.p)

        return loss

class CMD_Loss(nn.Module):
    def __init__(self, K=1, p=2, decay=1.0):
        super(CMD_Loss, self).__init__()
        assert(K>0)
        assert(p>0)
        assert(decay>0)
        self.K = K
        self.p = p
        self.decay = decay

    def forward(self, inputA, inputB):
        meanA = inputA.mean(0, keepdim=True)
        meanB = inputB.mean(0, keepdim=True)

        assert(meanA.size()==meanB.size())
        loss = meanA.sub(meanB).norm(self.p)

        if self.K>1:
            meanA = meanA.expand_as(inputA)
            meanB = meanB.expand_as(inputB)
            distA = inputA.sub(meanA)
            distB = inputB.sub(meanB)
            for i in range(2,self.K+1):
                CM_A = distA.pow(self.K).mean(0)
                CM_B = distB.pow(self.K).mean(0)
                loss_k = CM_A.sub(CM_B).norm(self.p)
                loss +=loss_k*pow(self.decay,i-1)

        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

