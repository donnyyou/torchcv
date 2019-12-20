#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualize the tensor of the computer vision.


import os
import cv2
import numpy as np

from lib.datasets.tools.transforms import DeNormalize
from lib.tools.util.logger import Logger as Log


TENSOR_DIR = 'vis/results/tensor'


class TensorVisualizer(object):

    def __init__(self, configer):
        self.configer = configer

    def vis_tensor(self, tensor, name='default', sub_dir=''):
        base_dir = os.path.join(self.configer.get('project_dir'), TENSOR_DIR, sub_dir)

        if not isinstance(tensor, np.ndarray):
            if len(tensor.size()) != 3:
                Log.error('Tensor size is not valid.')
                exit(1)

            tensor = tensor.data.cpu().numpy().transpose(1, 2, 0)

        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        tensor_img = cv2.resize(tensor, tuple(self.configer.get('data', 'input_size')))
        cv2.imwrite(tensor_img, os.path.join(base_dir, '{}.jpg'.format(name)))

    def vis_img(self, image_in, name='default', sub_dir='images'):
        base_dir = os.path.join(self.configer.get('project_dir'), TENSOR_DIR, sub_dir)

        if not isinstance(image_in, np.ndarray):
            if len(image_in.size()) != 3:
                Log.error('Image size is not valid.')
                exit(1)

            image = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                mean=self.configer.get('normalize', 'mean'),
                                std=self.configer.get('normalize', 'std'))(image_in.clone())
            image = image.data.cpu().numpy().transpose(1, 2, 0)
        else:
            image = image_in.copy()

        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        img = cv2.resize(image, tuple(self.configer.get('data', 'input_size')))
        cv2.imwrite(img, os.path.join(base_dir, '{}.jpg'.format(name)))


if __name__ == "__main__":
    # Test the visualizer.
    pass
