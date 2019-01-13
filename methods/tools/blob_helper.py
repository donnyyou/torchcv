#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Generate the inputs.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch

from datasets.tools.transforms import DeNormalize, ToTensor, Normalize
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class BlobHelper(object):
    def __init__(self, configer):
        self.configer = configer

    def make_input_batch(self, image_list, input_size=None, scale=1.0):
        input_list = list()
        for image in image_list:
            input_list.append(self.make_input(image, input_size=input_size, scale=scale))

        return torch.cat(input_list, 0)

    def make_input(self, image=None, input_size=None,
                   min_side_length=None, max_side_length=None, scale=None):
        if input_size is not None and min_side_length is None and max_side_length is None:
            if input_size[0] == -1 and input_size[1] == -1:
                in_width, in_height = ImageHelper.get_size(image)

            elif input_size[0] != -1 and input_size[1] != -1:
                in_width, in_height = input_size

            elif input_size[0] == -1 and input_size[1] != -1:
                width, height = ImageHelper.get_size(image)
                scale_ratio = input_size[1] / height
                w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
                in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

            else:
                assert input_size[0] != -1 and input_size[1] == -1
                width, height = ImageHelper.get_size(image)
                scale_ratio = input_size[0] / width
                w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
                in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

        elif input_size is None and min_side_length is not None and max_side_length is None:
            width, height = ImageHelper.get_size(image)
            scale_ratio = min_side_length / min(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

        elif input_size is None and min_side_length is None and max_side_length is not None:
            width, height = ImageHelper.get_size(image)
            scale_ratio = max_side_length / max(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

        elif input_size is None and min_side_length is not None and max_side_length is not None:
            width, height = ImageHelper.get_size(image)
            scale_ratio = min_side_length / min(width, height)
            bound_scale_ratio = max_side_length / max(width, height)
            scale_ratio = min(scale_ratio, bound_scale_ratio)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            in_width, in_height = int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))

        else:
            in_width, in_height = ImageHelper.get_size(image)

        image = ImageHelper.resize(image, (int(in_width * scale), int(in_height * scale)), interpolation='cubic')
        img_tensor = ToTensor()(image)
        img_tensor = Normalize(div_value=self.configer.get('normalize', 'div_value'),
                               mean=self.configer.get('normalize', 'mean'),
                               std=self.configer.get('normalize', 'std'))(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(torch.device('cpu' if self.configer.get('gpu') is None else 'cuda'))

        return img_tensor

    def tensor2bgr(self, tensor):
        assert len(tensor.size()) == 3

        ori_img = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                              mean=self.configer.get('normalize', 'mean'),
                              std=self.configer.get('normalize', 'std'))(tensor.cpu())
        ori_img = ori_img.numpy().transpose(1, 2, 0).astype(np.uint8)

        if self.configer.get('data', 'input_mode') == 'BGR':
            return ori_img
        else:
            image_bgr = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            return image_bgr
