#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Generate the inputs.


import cv2
import numpy as np
import torch

from lib.data.transforms import DeNormalize, ToTensor, Normalize
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.helper.dc_helper import DCHelper
from lib.tools.helper.tensor_helper import TensorHelper


class BlobHelper(object):
    def __init__(self, configer):
        self.configer = configer

    def get_blob(self, data_dict, scale=None, flip=False):
        assert scale is not None

        img_list, meta_list = [], []
        for image, meta in zip(DCHelper.tolist(data_dict['img']), DCHelper.tolist(data_dict['meta'])):
            c, h, w = image.size()
            border_hw = [int(h*scale), int(w*scale)]
            meta['border_wh'] = border_hw[::-1]
            image = TensorHelper.resize(image, border_hw, mode='bilinear', align_corners=True)
            if flip:
                image = image.flip([2])

            if self.configer.get('test.fit_stride', default=0) > 0:
                stride = self.configer.get('test', 'fit_stride')

                pad_w = 0 if (border_hw[1] % stride == 0) else stride - (border_hw[1] % stride)  # right
                pad_h = 0 if (border_hw[0] % stride == 0) else stride - (border_hw[0] % stride)  # down

                expand_image = torch.zeros((c, border_hw[0] + pad_h, border_hw[1] + pad_w)).to(image.device)
                expand_image[:, 0:border_hw[0], 0:border_hw[1]] = image
                image = expand_image

            img_list.append(image)
            meta_list.append(meta)

        new_data_dict = dict(
            img=DCHelper.todc(img_list, stack=True, samples_per_gpu=True),
            meta=DCHelper.todc(meta_list, samples_per_gpu=True, cpu_only=True)
        )
        return new_data_dict

    def tensor2bgr(self, tensor):
        assert len(tensor.size()) == 3

        ori_img = DeNormalize(**self.configer.get('data', 'normalize'))(tensor.cpu())
        ori_img = ori_img.numpy().transpose(1, 2, 0).astype(np.uint8)

        if self.configer.get('data', 'input_mode') == 'BGR':
            return ori_img
        else:
            image_bgr = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            return image_bgr
