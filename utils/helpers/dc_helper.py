#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import itertools

from extensions.tools.parallel import DataContainer


class DCHelper(object):

    @staticmethod
    def tolist(dc):
        return list(itertools.chain(*dc.data))

    @staticmethod
    def todc(data_list, samples_per_gpu, cpu_only=False):
        stacked = []
        for i in range(0, len(data_list), samples_per_gpu):
            stacked.append(data_list[i:i + samples_per_gpu])

        return DataContainer(stacked, cpu_only=cpu_only)
