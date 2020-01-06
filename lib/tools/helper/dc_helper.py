#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import itertools

from lib.parallel.data_container import DataContainer


class DCHelper(object):

    @staticmethod
    def tolist(dc):
        if isinstance(dc, (list, tuple)):
            return dc

        if isinstance(dc, torch.Tensor):
            return [item for item in dc]

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
    def todc(data_list, samples_per_gpu=True, stack=False, cpu_only=False, device_ids=None, concat=False):
        if not samples_per_gpu:
             if not stack:
                 return DataContainer(data_list, stack=stack, samples_per_gpu=samples_per_gpu, cpu_only=cpu_only)
             else:
                 return DataContainer(torch.stack(data_list, 0), stack=stack, samples_per_gpu=samples_per_gpu, cpu_only=cpu_only)

        device_ids = list(range(torch.cuda.device_count())) if device_ids is None else device_ids
        samples = (len(data_list) - 1 + len(device_ids)) // len(device_ids)
        stacked = []
        for i in range(0, len(data_list), samples):
            if not stack and not concat:
                stacked.append(data_list[i:i + samples])
            elif stack:
                stacked.append(torch.stack(data_list[i:i + samples], 0))
            else:
                stacked.append(torch.cat(data_list[i:i + samples], 0))

        return DataContainer(stacked, stack=stack, samples_per_gpu=samples_per_gpu, cpu_only=cpu_only)
