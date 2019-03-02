import torch
import torch.nn as nn
from torch.autograd import Variable


class CMD_K_Loss(nn.Module):
    def __init__(self, K=1, p=2):
        super(CMD_K_Loss, self).__init__()
        assert(K>0)
        assert(p>0)
        self.K=K
        self.p=p

    def forward(self, inputA, inputB):
        N_A = inputA.size(0)
        N_B = inputB.size(0)

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
        N_A = inputA.size(0)
        N_B = inputB.size(0)

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


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def get_target_tensor_new(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.Tensor(input.size()).fill_(self.real_label)
        else:
            target_tensor = self.Tensor(input.size()).fill_(self.fake_label)
        target_tensor = Variable(target_tensor, requires_grad=False)
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor_new(input, target_is_real)
        return self.loss(input, target_tensor)