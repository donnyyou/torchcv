import numpy as np
import torch

from extensions.nms.src.cpu_nms import cpu_nms
from extensions.nms.src.cpu_soft_nms import cpu_soft_nms
from extensions.nms.src.gpu_nms import gpu_nms


def nms(dets, thresh, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations."""

    if isinstance(dets, torch.Tensor):
        if dets.is_cuda:
            device_id = dets.get_device()

        dets = dets.detach().cpu().numpy()
    assert isinstance(dets, np.ndarray)

    if dets.shape[0] == 0:
        inds = []
    else:
        inds = (gpu_nms(dets, thresh, device_id=device_id)
                if device_id is not None else cpu_nms(dets, thresh))

    return np.array(inds, dtype=np.int)


def soft_nms(dets, max_threshold=0.3, method='linear', sigma=0.5, min_score=0):
    if isinstance(dets, torch.Tensor):
        _dets = dets.detach().cpu().numpy()
    else:
        _dets = dets.copy()
    assert isinstance(_dets, np.ndarray)

    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)

    new_dets, inds = cpu_soft_nms(
        _dets, Nt=max_threshold, method=methods[method], sigma=sigma, threshold=min_score)

    if isinstance(dets, torch.Tensor):
        return dets.new_tensor(
            inds, dtype=torch.long), dets.new_tensor(new_dets)
    else:
        return np.array(
            inds, dtype=np.int), np.array(
                new_dets, dtype=np.float32)
