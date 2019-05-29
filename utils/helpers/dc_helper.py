#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import itertools

from extensions.tools.parallel import DataContainer


class DCHelper(object):

    @staticmethod
    def tolist(dc):
        if isinstance(dc, (list, tuple)):
            return dc

        assert isinstance(dc, DataContainer), type(dc)
        if dc.samples_per_gpu and not dc.stack:
            return list(itertools.chain(*dc.data))
        elif dc.samples_per_gpu and dc.stack:
            return list(itertools.chain(*[[item for item in sub_batch] for sub_batch in dc.data]))
        else:
            return dc.data

    @staticmethod
    def totensor(dc):
        assert isinstance(dc, DataContainer), type(dc)
        if isinstance(dc.data, torch.Tensor):
            return dc.data

        assert isinstance(dc.data, (list, tuple)) and isinstance(dc.data[0], torch.Tensor)
        return torch.cat(dc.data, 0)

    @staticmethod
    def todc(data_list, samples_per_gpu, stack=False, cpu_only=False):
        stacked = []
        for i in range(0, len(data_list), samples_per_gpu):
            if not stack:
                stacked.append(data_list[i:i + samples_per_gpu])
            else:
                stacked.append(torch.cat(data_list[i:i + samples_per_gpu], 0))

        return DataContainer(stacked, stack=stack, samples_per_gpu=True, cpu_only=cpu_only)
