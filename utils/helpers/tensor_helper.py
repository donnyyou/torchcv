#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Repackage some image operations.


import torch.nn.functional as F


class TensorHelper(object):

    @staticmethod
    def resize(tensor, target_hw, mode=None, **kwargs):
        tensor_type = tensor.type()
        dim = len(tensor.size())
        for i in range(4 - dim):
            tensor = tensor.unsqueeze(0)

        tensor = tensor.float()
        tensor = F.interpolate(tensor.float(), target_hw, mode=mode, **kwargs)
        for i in range(4 - dim):
            tensor = tensor.squeeze(0)

        return  tensor.type(tensor_type)

