##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Data Parallel"""
import functools
import threading

import torch
import torch.cuda.comm as comm
from torch.autograd import Function
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import get_a_var
from torch.nn.parallel.scatter_gather import gather

from .scatter_gather import scatter_kwargs


torch_ver = torch.__version__[:3]


class Reduce(Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. "Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0, gather_=True):
        super(DataParallelModel, self).__init__(module, device_ids, output_device, dim)
        self.gather_ = gather_

    def gather(self, outputs, output_device):
        if self.gather_:
            return gather(outputs, output_device, dim=self.dim)

        return outputs

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.
    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.
    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
    Example::
        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelCriterion, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, inputs, *targets, gathered=True, **kwargs):
        # input should be already scatterd
        # scattering the targets instead
        if gathered:
            if isinstance(inputs, (list, tuple)):
                inputs, _ = self.scatter(inputs, kwargs, self.device_ids)
            else:
                inputs, _ = self.scatter([inputs], kwargs, self.device_ids)
                inputs = tuple(inputs_per_gpu[0] for inputs_per_gpu in inputs)

        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)

        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs[0], *targets[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        # targets = tuple(targets_per_gpu[0] for targets_per_gpu in targets)
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
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
    if torch_ver != "0.3":
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        if torch_ver != "0.3":
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(input, *target, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, target,
                                          kwargs, device),)
                   for i, (module, input, target, kwargs, device) in
                   enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], targets[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


###########################################################################
# Adapted from Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
#
class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created
    by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead
    of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate
