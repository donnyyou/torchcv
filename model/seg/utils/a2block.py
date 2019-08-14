#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai Li(lxtpku@pku.edu.cn)
# Pytorch Implementation of A2-net(nips 2018)

import torch
import torch.nn as nn


class A2Block(nn.Module):
    def __init__(self, inplane, plane):
        super(A2Block, self).__init__()
        self.down = nn.Conv2d(inplane, plane, 1)
        self.up = nn.Conv2d(plane, inplane, 1)
        self.gather_down = nn.Conv2d(inplane, plane, 1)
        self.distribue_down = nn.Conv2d(inplane, plane, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        res = x
        A = self.down(res)
        B = self.gather_down(res)
        b, c, h, w = A.size()
        A = A.view(b, c, -1)  # (b, c, h*w)
        B = B.view(b, c, -1)  # (b, c, h*w)
        B = self.softmax(B)
        B = B.permute(0, 2, 1)  # (b, h*w, c)

        G = torch.bmm(A, B)  # (b,c,c)

        C = self.distribue_down(res)
        C = C.view(b, c, -1)  # (b, c, h*w)
        C = self.softmax(C)
        C = C.permute(0, 2, 1)  # (b, h*w, c)

        atten = torch.bmm(C, G)  # (b, h*w, c)
        atten = atten.permute(0, 2, 1).view(b, c, h, -1)
        atten = self.up(atten)

        out = res + atten
        return out
