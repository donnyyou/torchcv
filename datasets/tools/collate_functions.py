#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (donnyyou@163.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.tools.logger import Logger as Log


NOT_STACK_KEYS = ['kpts', 'bboxes', 'labels', 'polygons']


class CollateFunctions(object):

    @staticmethod
    def our_collate(batch, data_keys=None, trans_dict=None):
        transposed = [list(sample) for sample in zip(*batch)]
        data_dict = CollateFunctions.trans(data_keys, transposed, trans_dict)
        return data_dict

    @staticmethod
    def default_collate(batch, data_keys=None):
        transposed = [list(sample) for sample in zip(*batch)]
        data_dict = {key: CollateFunctions.stack(value) for key, value in zip(data_keys, transposed)}
        return data_dict

    @staticmethod
    def stack(batch):
        if isinstance(batch, torch.Tensor) or isinstance(batch[0], (list, tuple)):
            return batch

        elif isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, 0)

        elif isinstance(batch[0], np.ndarray):
            return torch.stack([torch.from_numpy(b) for b in batch], 0)

        elif isinstance(batch[0], Image.Image):
            return torch.stack([torch.from_numpy(np.array(b)) for b in batch], 0)

        elif isinstance(batch[0], int):
            return torch.LongTensor(batch)

        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)

        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        raise TypeError((error_msg.format(type(batch[0]))))

    @staticmethod
    def trans(data_keys, out_list, trans_dict):
        if trans_dict['size_mode'] == 'random_size':
            return {key: CollateFunctions.stack(value) if key not in NOT_STACK_KEYS else value
                    for key, value in zip(data_keys, out_list)}

        img_list = out_list[data_keys.index('img')]

        if trans_dict['size_mode'] == 'fix_size':
            target_width, target_height = trans_dict['input_size']

        elif trans_dict['size_mode'] == 'stride_size':
            w = img_list[0].size(2)
            h = img_list[0].size(1)
            pad_w = 0 if (w % trans_dict['stride'] == 0) else trans_dict['stride'] - (w % trans_dict['stride'])  # right
            pad_h = 0 if (h % trans_dict['stride'] == 0) else trans_dict['stride'] - (h % trans_dict['stride'])  # down
            target_width = w + pad_w
            target_height = h + pad_h

        elif trans_dict['size_mode'] == 'multi_size':
            ms_input_size = trans_dict['ms_input_size']
            target_width, target_height = ms_input_size[random.randint(0, len(ms_input_size) - 1)]

        elif trans_dict['size_mode'] == 'max_size':
            border_width = [img.size(2) for img in img_list]
            border_height = [img.size(1) for img in img_list]
            target_width, target_height = max(border_width), max(border_height)

        else:
            raise NotImplementedError('Size Mode {} is invalid!'.format(trans_dict['size_mode']))

        for i in range(len(img_list)):
            channels, height, width = img_list[i].size()
            if height == target_height and width == target_width:
                continue

            scaled_size = [width, height]

            if trans_dict['align_method'] in ['only_scale', 'scale_and_pad']:
                w_scale_ratio = target_width / width
                h_scale_ratio = target_height / height
                if trans_dict['align_method'] == 'scale_and_pad':
                    w_scale_ratio = min(w_scale_ratio, h_scale_ratio)
                    h_scale_ratio = w_scale_ratio

                if 'kpts' in data_keys and out_list[data_keys.index('kpts')][i].numel() > 0:
                    out_list[data_keys.index('kpts')][i][:, :, 0] *= w_scale_ratio
                    out_list[data_keys.index('kpts')][i][:, :, 1] *= h_scale_ratio

                if 'bboxes' in data_keys and out_list[data_keys.index('bboxes')][i].numel() > 0:
                    out_list[data_keys.index('bboxes')][i][:, 0::2] *= w_scale_ratio
                    out_list[data_keys.index('bboxes')][i][:, 1::2] *= h_scale_ratio

                if 'polygons' in data_keys:
                    for object_id in range(len(out_list[data_keys.index('polygons')][i])):
                        for polygon_id in range(len(out_list[data_keys.index('polygons')][i][object_id])):
                            out_list[data_keys.index('polygons')][i][object_id][polygon_id][0::2] *= w_scale_ratio
                            out_list[data_keys.index('polygons')][i][object_id][polygon_id][1::2] *= h_scale_ratio

                scaled_size = (int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio)))
                img_list[i] = F.interpolate(img_list[i].unsqueeze(0),
                                            scaled_size, mode='bilinear', align_corners=False).squeeze(0)
                if 'labelmap' in data_keys:
                    labelmap = out_list[data_keys.index('labelmap')][i].unsqueeze(0).unsqueeze(0).float()
                    labelmap = F.interpolate(labelmap, scaled_size, mode='nearest')
                    labelmap = labelmap.long().squeeze(0).squeeze(0)
                    out_list[data_keys.index('labelmap')][i] = labelmap

                if 'maskmap' in data_keys:
                    maskmap = out_list[data_keys.index('maskmap')][i].unsqueeze(0).unsqueeze(0).float()
                    maskmap = F.interpolate(maskmap, scaled_size, mode='nearest')
                    maskmap = maskmap.long().squeeze(0).squeeze(0)
                    out_list[data_keys.index('maskmap')][i] = maskmap

            pad_width = target_width - scaled_size[0]
            pad_height = target_height - scaled_size[1]
            assert pad_height >= 0 and pad_width >= 0
            if pad_width > 0 or pad_height > 0:
                assert trans_dict['align_method'] in ['only_pad', 'scale_and_pad']
                left_pad = 0
                up_pad = 0
                if 'pad_mode' not in trans_dict or trans_dict['pad_mode'] == 'random':
                    left_pad = random.randint(0, pad_width)  # pad_left
                    up_pad = random.randint(0, pad_height)  # pad_up

                elif trans_dict['pad_mode'] == 'pad_left_up':
                    left_pad = pad_width
                    up_pad = pad_height

                elif trans_dict['pad_mode'] == 'pad_right_down':
                    left_pad = 0
                    up_pad = 0

                elif trans_dict['pad_mode'] == 'pad_center':
                    left_pad = pad_width // 2
                    up_pad = pad_height // 2

                else:
                    Log.error('Invalid pad mode: {}'.format(trans_dict['pad_mode']))
                    exit(1)

                pad = (left_pad, pad_width-left_pad, up_pad, pad_height-up_pad)

                img_list[i] = F.pad(img_list[i], pad=pad, value=0)

                if 'labelmap' in data_keys:
                    labelmap = out_list[data_keys.index('labelmap')][i]
                    out_list[data_keys.index('labelmap')][i] = F.pad(labelmap, pad=pad, value=-1)

                if 'maskmap' in data_keys:
                    maskmap = out_list[data_keys.index('maskmap')][i]
                    out_list[data_keys.index('maskmap')][i] = F.pad(maskmap, pad=pad, value=1)

                if 'polygons' in data_keys:
                    for object_id in range(len(out_list[data_keys.index('polygons')][i])):
                        for polygon_id in range(len(out_list[data_keys.index('polygons')][i][object_id])):
                            out_list[data_keys.index('polygons')][i][object_id][polygon_id][0::2] += left_pad
                            out_list[data_keys.index('polygons')][i][object_id][polygon_id][1::2] += up_pad

                if 'kpts' in data_keys and out_list[data_keys.index('kpts')][i].numel() > 0:
                    out_list[data_keys.index('kpts')][i][:, :, 0] += left_pad
                    out_list[data_keys.index('kpts')][i][:, :, 1] += up_pad

                if 'bboxes' in data_keys and out_list[data_keys.index('bboxes')][i].numel() > 0:
                    out_list[data_keys.index('bboxes')][i][:, 0::2] += left_pad
                    out_list[data_keys.index('bboxes')][i][:, 1::2] += up_pad

        return {key: CollateFunctions.stack(value) if key not in NOT_STACK_KEYS else value
                for key, value in zip(data_keys, out_list)}