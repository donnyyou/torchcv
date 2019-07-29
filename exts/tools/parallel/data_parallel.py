#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Adapted from: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/parallel.py


import functools
import threading

import torch
import torch.cuda.comm as comm
from torch.autograd import Function
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import get_a_var
from torch.nn.parallel.scatter_gather import gather
try:
    from torch._six import container_abcs
except:
    print("torch._six ImportError: Lower version of pytorch.")

from .scatter_gather import scatter_kwargs


class Reduce(Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


class ParallelModel(DataParallel):
    """
    Example::
        >>> net = ParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0, gather_=True):
        super(ParallelModel, self).__init__(module, device_ids, output_device, dim)
        self.gather_ = gather_

    def gather(self, outputs, output_device):
        if self.gather_:
            return gather(outputs, output_device, dim=self.dim)

        return outputs

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


class ParallelCriterion(DataParallel):
    """
    Example::
        >>> net = ParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = ParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(ParallelCriterion, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, inputs, **kwargs):
        # input should be already scatterd
        # scattering the targets instead
        if not self.device_ids:
            return self.module(inputs, **kwargs)

        kwargs = (kwargs, ) * len(inputs)
        if len(self.device_ids) == 1:
            return self.module(inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        # targets = tuple(targets_per_gpu[0] for targets_per_gpu in targets)
        outputs = _criterion_parallel_apply(replicas, inputs, kwargs)
        if isinstance(outputs[0], container_abcs.Mapping):
            return {key: (Reduce.apply(*[d[key] for d in outputs]) / len(outputs)) for key in outputs[0]}

        elif isinstance(outputs[0], container_abcs.Sequence):
            transposed = zip(*outputs)
            return [Reduce.apply(*samples) / len(outputs) for samples in transposed]

        else:
            return Reduce.apply(*outputs) / len(outputs)


def _criterion_parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(input, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device),)
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs
