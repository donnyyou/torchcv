import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class FC_Discriminator(nn.Module):
    def __init__(self, input_size,output_size, num_hidden=128, gpu_ids=[]):
        super(FC_Discriminator,self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, num_hidden),
            nn.Linear(num_hidden, output_size))
        self.gpu_ids = gpu_ids

    def forward(self, x):
        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):
            x = F.dropout(x, training=self.training)
            x = x.view(x.size(0), -1)
            x = nn.parallel.data_parallel(self.fc, x, self.gpu_ids)
            x = nn.parallel.data_parallel(torch.nn.functional.sigmoid, x, self.gpu_ids)
        else:
            x = F.dropout(x, training=self.training)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = torch.nn.functional.sigmoid(x)
        return x