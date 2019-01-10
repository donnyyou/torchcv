#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from extensions.parallel.data_container import DataContainer


class DCHelper(object):

    @staticmethod
    def tolist(dc):
        return list(itertools.chain(*dc.data))

    @staticmethod
    def todc(data_list, gpu_list, cpu_only=False):
        assert len(data_list) % len(gpu_list) == 0
        samples_per_gpu = len(data_list) // len(gpu_list)
        stacked = []
        for i in range(0, len(data_list), samples_per_gpu):
            stacked.append(data_list[i:i + samples_per_gpu])

        return DataContainer(stacked, cpu_only=cpu_only)
