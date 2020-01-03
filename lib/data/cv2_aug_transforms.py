#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Image Augmentations implemented by OpenCV. Including RandomPad, RandomRotate, RandomResize etc.


import collections
import random
import math
import cv2
import numpy as np
import imgaug.augmenters as iaa


class RandomBlur(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.blur_list = [
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AverageBlur(k=(2, 11)),
            iaa.AverageBlur(k=((5, 11), (1, 3))),
            iaa.MedianBlur(k=(3, 11)),
        ]

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, (np.ndarray, list))
        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        method = random.randint(0, len(self.blur_list)-1)
        img = self.blur_list[method].augment_image(img)
        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomErase(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, ratio=0.5, erase_range=(0.02, 0.4), aspect=0.3, mean=[104, 117, 123]):
        self.ratio = ratio
        self.mean = mean
        self.erase_range = erase_range
        self.aspect = aspect

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, (np.ndarray, list))
        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        height, width, channels = img.shape
        for _ in range(100):
            area = height * width
            target_area = random.uniform(self.erase_range[0], self.erase_range[1]) * area
            aspect_ratio = random.uniform(self.aspect, 1 / self.aspect)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = random.randint(0, height - h)
                y1 = random.randint(0, width - w)
                if channels == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomPad(object):
    """Random Pad a ``np.ndarray``

    Args:
        inputs: All elements that need to be processed.
        up_scale_range: (list): the padding scale range of the image.
        mean: (list): the mean pixel value.
        ratio: the ratio of random pad.

    Returns:
        Outputs: All elements that have been processed.
    """
    def __init__(self, up_scale_range=None, ratio=0.5, mean=(104, 117, 123)):
        assert isinstance(up_scale_range, (list, tuple))
        self.up_scale_range = up_scale_range
        self.ratio = ratio
        self.mean = tuple(mean)

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, (np.ndarray, list))
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)
        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        height, width, channels = img.shape if isinstance(img, np.ndarray) else img[0].shape
        ws = random.uniform(self.up_scale_range[0], self.up_scale_range[1])
        hs = ws
        for _ in range(50):
            scale = random.uniform(self.up_scale_range[0], self.up_scale_range[1])
            min_ratio = max(0.5, 1. / scale / scale)
            max_ratio = min(2, scale * scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            ws = scale * ratio
            hs = scale / ratio
            if ws >= 1 and hs >= 1:
                break

        pad_width = random.randint(0, int(ws * width) - width)
        pad_height = random.randint(0, int(hs * height) - height)
        left_pad = random.randint(0, pad_width)  # pad_left
        up_pad = random.randint(0, pad_height)  # pad_up
        if not isinstance(img, list):
            img = cv2.copyMakeBorder(img, up_pad, pad_height-up_pad, left_pad, pad_width-left_pad,
                                     cv2.BORDER_CONSTANT, value=self.mean)
        else:
            img = [cv2.copyMakeBorder(item, up_pad, pad_height-up_pad, left_pad, pad_width-left_pad,
                                      cv2.BORDER_CONSTANT, value=self.mean) for item in img]

        if labelmap is not None:
            labelmap = cv2.copyMakeBorder(labelmap, up_pad, pad_height - up_pad, left_pad, pad_width - left_pad,
                                          cv2.BORDER_CONSTANT, value=255)

        if maskmap is not None:
            maskmap = cv2.copyMakeBorder(maskmap, up_pad, pad_height - up_pad, left_pad, pad_width - left_pad,
                                         cv2.BORDER_CONSTANT, value=1)

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    polygons[object_id][polygon_id][0::2] += left_pad
                    polygons[object_id][polygon_id][1::2] += up_pad

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] += left_pad
            kpts[:, :, 1] += up_pad

        if bboxes is not None and bboxes.size > 0:
            bboxes[:, 0::2] += left_pad
            bboxes[:, 1::2] += up_pad

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomBorder(object):
    """ Padding the Image to proper size.
            Args:
                stride: the stride of the network.
                pad_value: the value that pad to the image border.
                img: Image object as input.
            Returns::
                img: Image object.
    """
    def __init__(self, pad=None, ratio=0.5, mean=(104, 117, 123), allow_outside_center=True):
        self.pad = pad
        self.ratio = ratio
        self.mean = mean
        self.allow_outside_center = allow_outside_center

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, (np.ndarray, list))
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        height, width, channels = img.shape if isinstance(img, np.ndarray) else img[0].shape
        left_pad, up_pad, right_pad, down_pad = self.pad
        target_size = [width + left_pad + right_pad, height + up_pad + down_pad]
        offset_left = -left_pad
        offset_up = -up_pad

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] -= offset_left
            kpts[:, :, 1] -= offset_up
            mask = np.logical_or.reduce((kpts[:, :, 0] >= target_size[0], kpts[:, :, 0] < 0,
                                         kpts[:, :, 1] >= target_size[1], kpts[:, :, 1] < 0))
            kpts[mask == 1, 2] = -1

        if bboxes is not None and bboxes.size > 0:
            if self.allow_outside_center:
                mask = np.ones(bboxes.shape[0], dtype=bool)
            else:
                crop_bb = np.array([offset_left, offset_up, offset_left + target_size[0], offset_up + target_size[1]])
                center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]).all(axis=1)

            bboxes[:, 0::2] -= offset_left
            bboxes[:, 1::2] -= offset_up
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, target_size[0] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, target_size[1] - 1)

            mask = np.logical_and(mask, (bboxes[:, :2] < bboxes[:, 2:]).all(axis=1))
            bboxes = bboxes[mask]
            if labels is not None:
                labels = labels[mask]

            if polygons is not None:
                new_polygons = list()
                for object_id in range(len(polygons)):
                    if mask[object_id] == 1:
                        for polygon_id in range(len(polygons[object_id])):
                            polygons[object_id][polygon_id][0::2] -= offset_left
                            polygons[object_id][polygon_id][1::2] -= offset_up
                            polygons[object_id][polygon_id][0::2] = np.clip(polygons[object_id][polygon_id][0::2],
                                                                            0, target_size[0] - 1)
                            polygons[object_id][polygon_id][1::2] = np.clip(polygons[object_id][polygon_id][1::2],
                                                                            0, target_size[1] - 1)

                        new_polygons.append(polygons[object_id])

                polygons = new_polygons

        if not isinstance(img, list):
            expand_image = np.zeros((max(height, target_size[1]) + abs(offset_up),
                                     max(width, target_size[0]) + abs(offset_left), channels), dtype=img.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
                         abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = img
            img = expand_image[max(offset_up, 0):max(offset_up, 0) + target_size[1],
                               max(offset_left, 0):max(offset_left, 0) + target_size[0]]
        else:
            for i in range(len(img)):
                expand_image = np.zeros((max(height, target_size[1]) + abs(offset_up),
                                         max(width, target_size[0]) + abs(offset_left), channels), dtype=img[i].dtype)
                expand_image[:, :, :] = self.mean
                expand_image[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
                             abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = img[i]
                img[i] = expand_image[max(offset_up, 0):max(offset_up, 0) + target_size[1],
                                      max(offset_left, 0):max(offset_left, 0) + target_size[0]]

        if maskmap is not None:
            expand_maskmap = np.zeros((max(height, target_size[1]) + abs(offset_up),
                                       max(width, target_size[0]) + abs(offset_left)), dtype=maskmap.dtype)
            expand_maskmap[:, :] = 1
            expand_maskmap[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
            abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = maskmap
            maskmap = expand_maskmap[max(offset_up, 0):max(offset_up, 0) + target_size[1],
                      max(offset_left, 0):max(offset_left, 0) + target_size[0]]

        if labelmap is not None:
            expand_labelmap = np.zeros((max(height, target_size[1]) + abs(offset_up),
                                        max(width, target_size[0]) + abs(offset_left)), dtype=labelmap.dtype)
            expand_labelmap[:, :] = 255
            expand_labelmap[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
            abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = labelmap
            labelmap = expand_labelmap[max(offset_up, 0):max(offset_up, 0) + target_size[1],
                       max(offset_left, 0):max(offset_left, 0) + target_size[0]]

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomHFlip(object):
    def __init__(self, swap_pair=None, ratio=0.5):
        self.swap_pair = swap_pair
        self.ratio = ratio

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, (np.ndarray, list))
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        # height, width, _ = img.shape
        height, width, _ = img.shape if isinstance(img, np.ndarray) else img[0].shape
        if not isinstance(img, list):
            img = cv2.flip(img, 1)
        else:
            img = [cv2.flip(item, 1) for item in img]

        if labelmap is not None:
            labelmap = cv2.flip(labelmap, 1)
            for pair in self.swap_pair:
                a_mask = (labelmap == pair[0])
                labelmap[labelmap == pair[1]] = pair[0]
                labelmap[a_mask] = pair[1]

        if maskmap is not None:
            maskmap = cv2.flip(maskmap, 1)

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    polygons[object_id][polygon_id][0::2] = width - 1 - polygons[object_id][polygon_id][0::2]

        if bboxes is not None and bboxes.size > 0:
            xmin = width - 1 - bboxes[:, 2]
            xmax = width - 1 - bboxes[:, 0]
            bboxes[:, 0] = xmin
            bboxes[:, 2] = xmax

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] = width - 1 - kpts[:, :, 0]

            for pair in self.swap_pair:
                temp_point = np.copy(kpts[:, pair[0] - 1])
                kpts[:, pair[0] - 1] = kpts[:, pair[1] - 1]
                kpts[:, pair[1] - 1] = temp_point

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = ratio
        assert self.upper >= self.lower, "saturation upper must be >= lower."
        assert self.lower >= 0, "saturation lower must be non-negative."

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] *= random.uniform(self.lower, self.upper)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomHue(object):
    def __init__(self, delta=18, ratio=0.5):
        assert 0 <= delta <= 360
        self.delta = delta
        self.ratio = ratio

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] += random.uniform(-self.delta, self.delta)
        img[:, :, 0][img[:, :, 0] > 360] -= 360
        img[:, :, 0][img[:, :, 0] < 0] += 360
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomPerm(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        swap = self.perms[random.randint(0, len(self.perms) - 1)]
        img = img[:, :, swap].astype(np.uint8)
        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = ratio
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)
        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        img = img.astype(np.float32)
        img *= random.uniform(self.lower, self.upper)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomBrightness(object):
    def __init__(self, shift_value=30, ratio=0.5):
        self.shift_value = shift_value
        self.ratio = ratio

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)
        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, crop_size, scale_range=(0.08, 1.0), aspect_range=(3. / 4., 4. / 3.)):
        self.size = tuple(crop_size)
        self.scale = scale_range
        self.ratio = aspect_range

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        height, width, _ = img.shape
        for attempt in range(10):
            area = width * height
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback
        w = min(height, width)
        i = (height - w) // 2
        j = (width - w) // 2
        return i, j, w, w

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        """
        Args:
            img (Numpy Image): Image to be cropped and resized.

        Returns:
            Numpy Image: Randomly cropped and resized image.
        """
        assert labelmap is None and maskmap is None and kpts is None and bboxes is None and labels is None
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = img[i:i+h, j:j+w]
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomResize(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """
    def __init__(self, scale_range=(0.75, 1.25), aspect_range=(0.9, 1.1), target_size=None,
                 resize_bound=None, method='random', ratio=0.5):
        self.scale_range = scale_range
        self.aspect_range = aspect_range
        self.resize_bound = resize_bound
        self.method = method
        self.ratio = ratio

        if target_size is not None:
            if isinstance(target_size, int):
                self.input_size = (target_size, target_size)
            elif isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                self.input_size = target_size
            else:
                raise TypeError('Got inappropriate size arg: {}'.format(target_size))
        else:
            self.input_size = None

    def get_scale(self, img_size, bboxes):
        if self.method == 'random':
            scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1])
            return scale_ratio

        elif self.method == 'focus':
            if self.input_size is not None and bboxes is not None and len(bboxes) > 0:
                bboxes = np.array(bboxes)
                border = bboxes[:, 2:] - bboxes[:, 0:2]
                scale = 0.6 / max(max(border[:, 0]) / self.input_size[0], max(border[:, 1]) / self.input_size[1])
                scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1]) * scale
                return scale_ratio

            else:
                scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1])
                return scale_ratio

        elif self.method == 'bound':
            scale1 = self.resize_bound[0] / min(img_size)
            scale2 = self.resize_bound[1] / max(img_size)
            scale = min(scale1, scale2)
            return scale

        else:
            raise NotImplementedError('Resize method {} undefined!'.format(self.method))

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        """
        Args:
            img     (Image):   Image to be resized.
            maskmap    (Image):   Mask to be resized.
            kpt     (list):    keypoints to be resized.
            center: (list):    center points to be resized.

        Returns:
            Image:  Randomly resize image.
            Image:  Randomly resize maskmap.
            list:   Randomly resize keypoints.
            list:   Randomly resize center points.
        """
        assert isinstance(img, (np.ndarray, list))
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)
        height, width, _ = img.shape if isinstance(img, np.ndarray) else img[0].shape
        if random.random() < self.ratio:
            scale_ratio = self.get_scale([width, height], bboxes)
            aspect_ratio = random.uniform(*self.aspect_range)
            w_scale_ratio = math.sqrt(aspect_ratio) * scale_ratio
            h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_ratio
        else:
            w_scale_ratio, h_scale_ratio = 1.0, 1.0

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] *= w_scale_ratio
            kpts[:, :, 1] *= h_scale_ratio

        if bboxes is not None and bboxes.size > 0:
            bboxes[:, 0::2] *= w_scale_ratio
            bboxes[:, 1::2] *= h_scale_ratio

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    polygons[object_id][polygon_id][0::2] *= w_scale_ratio
                    polygons[object_id][polygon_id][1::2] *= h_scale_ratio

        converted_size = (int(width * w_scale_ratio), int(height * h_scale_ratio))
        if not isinstance(img, list):
            img = cv2.resize(img, converted_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        else:
            img = [cv2.resize(item, converted_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8) for item in img]

        if labelmap is not None:
            labelmap = cv2.resize(labelmap, converted_size, interpolation=cv2.INTER_NEAREST)

        if maskmap is not None:
            maskmap = cv2.resize(maskmap, converted_size, interpolation=cv2.INTER_NEAREST)

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomRotate(object):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """
    def __init__(self, max_degree, ratio=0.5, mean=(104, 117, 123)):
        assert isinstance(max_degree, int)
        self.max_degree = max_degree
        self.ratio = ratio
        self.mean = mean

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        """
        Args:
            img    (Image):     Image to be rotated.
            maskmap   (Image):     Mask to be rotated.
            kpt    (list):      Keypoints to be rotated.
            center (list):      Center points to be rotated.

        Returns:
            Image:     Rotated image.
            list:      Rotated key points.
        """
        assert isinstance(img, (np.ndarray, list))
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)
        if random.random() < self.ratio:
            rotate_degree = random.uniform(-self.max_degree, self.max_degree)
        else:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        height, width, _ = img.shape if isinstance(img, np.ndarray) else img[0].shape
        img_center = (width / 2.0, height / 2.0)
        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]
        if not isinstance(img, list):
            img = cv2.warpAffine(img, rotate_mat, (new_width, new_height), borderValue=self.mean).astype(np.uint8)
        else:
            img = [cv2.warpAffine(item, rotate_mat, (new_width, new_height),
                                  borderValue=self.mean).astype(np.uint8) for item in img]
        if labelmap is not None:
            labelmap = cv2.warpAffine(labelmap, rotate_mat, (new_width, new_height),
                                      borderValue=(255, 255, 255), flags=cv2.INTER_NEAREST)
            labelmap = labelmap.astype(np.uint8)

        if maskmap is not None:
            maskmap = cv2.warpAffine(maskmap, rotate_mat, (new_width, new_height),
                                     borderValue=(1, 1, 1), flags=cv2.INTER_NEAREST)
            maskmap = maskmap.astype(np.uint8)

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    for i in range(len(polygons[object_id][polygon_id]) // 2):
                        x = polygons[object_id][polygon_id][i * 2]
                        y = polygons[object_id][polygon_id][i * 2 + 1]
                        p = np.array([x, y, 1])
                        p = rotate_mat.dot(p)
                        polygons[object_id][polygon_id][i * 2] = p[0]
                        polygons[object_id][polygon_id][i * 2 + 1] = p[1]

        if kpts is not None and kpts.size > 0:
            num_objects = len(kpts)
            num_keypoints = len(kpts[0])
            for i in range(num_objects):
                for j in range(num_keypoints):
                    x = kpts[i][j][0]
                    y = kpts[i][j][1]
                    p = np.array([x, y, 1])
                    p = rotate_mat.dot(p)
                    kpts[i][j][0] = p[0]
                    kpts[i][j][1] = p[1]

        # It is not right for object detection tasks.
        if bboxes is not None and bboxes.size > 0:
            for i in range(len(bboxes)):
                bbox_temp = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][1],
                             bboxes[i][0], bboxes[i][3], bboxes[i][2], bboxes[i][3]]

                for node in range(4):
                    x = bbox_temp[node * 2]
                    y = bbox_temp[node * 2 + 1]
                    p = np.array([x, y, 1])
                    p = rotate_mat.dot(p)
                    bbox_temp[node * 2] = p[0]
                    bbox_temp[node * 2 + 1] = p[1]

                bboxes[i] = [min(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                             min(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7]),
                             max(bbox_temp[0], bbox_temp[2], bbox_temp[4], bbox_temp[6]),
                             max(bbox_temp[1], bbox_temp[3], bbox_temp[5], bbox_temp[7])]

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomCrop(object):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int or tuple): Desired output size of the crop.(w, h)
    """

    def __init__(self, crop_size, ratio=0.5, method='random', grid=None, allow_outside_center=True):
        self.ratio = ratio
        self.method = method
        self.grid = grid
        self.allow_outside_center = allow_outside_center
        if isinstance(crop_size, float):
            self.size = (crop_size, crop_size)
        elif isinstance(crop_size, collections.Iterable) and len(crop_size) == 2:
            self.size = crop_size
        else:
            raise TypeError('Got inappropriate size arg: {}'.format(crop_size))

    def get_lefttop(self, crop_size, img_size):
        if self.method == 'center':
            return [(img_size[0] - crop_size[0]) // 2, (img_size[1] - crop_size[1]) // 2]

        elif self.method == 'random':
            x = random.randint(0, img_size[0] - crop_size[0])
            y = random.randint(0, img_size[1] - crop_size[1])
            return [x, y]

        elif self.method == 'grid':
            grid_x = random.randint(0, self.grid[0] - 1)
            grid_y = random.randint(0, self.grid[1] - 1)
            x = grid_x * ((img_size[0] - crop_size[0]) // (self.grid[0] - 1))
            y = grid_y * ((img_size[1] - crop_size[1]) // (self.grid[1] - 1))
            return [x, y]

        else:
            raise NotImplementedError('Random Crop Method {} Undefined!'.format(self.method))

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        """
        Args:
            img (Image):   Image to be cropped.
            maskmap (Image):  Mask to be cropped.

        Returns:
            Image:  Cropped image.
            Image:  Cropped maskmap.
            list:   Cropped keypoints.
            list:   Cropped center points.
        """
        assert isinstance(img, (np.ndarray, list))
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)
        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        height, width, _ = img.shape if isinstance(img, np.ndarray) else img[0].shape
        target_size = [min(self.size[0], width), min(self.size[1], height)]
        offset_left, offset_up = self.get_lefttop(target_size, [width, height])
        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] -= offset_left
            kpts[:, :, 1] -= offset_up

        if bboxes is not None and bboxes.size > 0:
            if self.allow_outside_center:
                mask = np.ones(bboxes.shape[0], dtype=bool)
            else:
                crop_bb = np.array([offset_left, offset_up, offset_left + target_size[0], offset_up + target_size[1]])
                center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]).all(axis=1)

            bboxes[:, 0::2] -= offset_left
            bboxes[:, 1::2] -= offset_up
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, target_size[0] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, target_size[1] - 1)
            mask = np.logical_and(mask, (bboxes[:, :2] < bboxes[:, 2:]).all(axis=1))
            bboxes = bboxes[mask]
            if labels is not None:
                labels = labels[mask]

            if polygons is not None:
                new_polygons = list()
                for object_id in range(len(polygons)):
                    if mask[object_id] == 1:
                        for polygon_id in range(len(polygons[object_id])):

                            polygons[object_id][polygon_id][0::2] -= offset_left
                            polygons[object_id][polygon_id][1::2] -= offset_up
                            polygons[object_id][polygon_id][0::2] = np.clip(polygons[object_id][polygon_id][0::2],
                                                                            0, target_size[0] - 1)
                            polygons[object_id][polygon_id][1::2] = np.clip(polygons[object_id][polygon_id][1::2],
                                                                            0, target_size[1] - 1)

                        new_polygons.append(polygons[object_id])

                polygons = new_polygons

        if not isinstance(img, list):
            img = img[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]
        else:
            img = [item[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]] for item in img]

        if maskmap is not None:
            maskmap = maskmap[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]

        if labelmap is not None:
            labelmap = labelmap[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomFocusCrop(object):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int or tuple): Desired output size of the crop.(w, h)
    """
    def __init__(self, crop_size, ratio=0.5, center_jitter=None, mean=(104, 117, 123), allow_outside_center=True):
        self.ratio = ratio
        self.center_jitter = center_jitter
        self.mean = mean
        self.allow_outside_center = allow_outside_center
        if isinstance(crop_size, float):
            self.size = (crop_size, crop_size)
        elif isinstance(crop_size, collections.Iterable) and len(crop_size) == 2:
            self.size = crop_size
        else:
            raise TypeError('Got inappropriate size arg: {}'.format(crop_size))

    def get_center(self, img_size, bboxes):
        if bboxes is None or bboxes.size == 0:
            if img_size[0] > self.size[0]:
                x = random.randint(self.size[0] // 2, img_size[0] - self.size[0] // 2)
            else:
                x = img_size[0] // 2

            if img_size[1] > self.size[1]:
                y = random.randint(self.size[1] // 2, img_size[1] - self.size[1] // 2)
            else:
                y = img_size[1] // 2

            return [x, y], -1

        else:
            border = bboxes[:, 2:] - bboxes[:, 0:2]
            area = border[:, 0] * border[:, 1]
            max_index = np.argmax(area)
            max_center = [(bboxes[max_index][0] + bboxes[max_index][2]) / 2,
                          (bboxes[max_index][1] + bboxes[max_index][3]) / 2]

            if self.center_jitter is not None:
                jitter = random.randint(-self.center_jitter, self.center_jitter)
                max_center[0] += jitter
                jitter = random.randint(-self.center_jitter, self.center_jitter)
                max_center[1] += jitter

            return max_center, max_index

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        """
        Args:
            img (Image):   Image to be cropped.
            maskmap (Image):  Mask to be cropped.

        Returns:
            Image:  Cropped image.
            Image:  Cropped maskmap.
            list:   Cropped keypoints.
            list:   Cropped center points.
        """
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        height, width, channels = img.shape

        center, index = self.get_center([width, height], bboxes)

        # img = ImageHelper.draw_box(img, bboxes[index])
        offset_left = int(center[0] - self.size[0] // 2)
        offset_up = int(center[1] - self.size[1] // 2)

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] -= offset_left
            kpts[:, :, 1] -= offset_up
            mask = np.logical_or.reduce((kpts[:, :, 0] >= self.size[0], kpts[:, :, 0] < 0,
                                         kpts[:, :, 1] >= self.size[1], kpts[:, :, 1] < 0))
            kpts[mask == 1, 2] = -1

        if bboxes is not None and bboxes.size > 0:
            if self.allow_outside_center:
                mask = np.ones(bboxes.shape[0], dtype=bool)
            else:
                crop_bb = np.array([offset_left, offset_up, offset_left + self.size[0], offset_up + self.size[1]])
                center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]).all(axis=1)

            bboxes[:, 0::2] -= offset_left
            bboxes[:, 1::2] -= offset_up
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.size[0] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.size[1] - 1)

            mask = np.logical_and(mask, (bboxes[:, :2] < bboxes[:, 2:]).all(axis=1))
            bboxes = bboxes[mask]
            if labels is not None:
                labels = labels[mask]

            if polygons is not None:
                new_polygons = list()
                for object_id in range(len(polygons)):
                    if mask[object_id] == 1:
                        for polygon_id in range(len(polygons[object_id])):
                            polygons[object_id][polygon_id][0::2] -= offset_left
                            polygons[object_id][polygon_id][1::2] -= offset_up
                            polygons[object_id][polygon_id][0::2] = np.clip(polygons[object_id][polygon_id][0::2],
                                                                            0, self.size[0] - 1)
                            polygons[object_id][polygon_id][1::2] = np.clip(polygons[object_id][polygon_id][1::2],
                                                                            0, self.size[1] - 1)

                        new_polygons.append(polygons[object_id])

                polygons = new_polygons

        expand_image = np.zeros((max(height, self.size[1]) + abs(offset_up),
                                 max(width, self.size[0]) + abs(offset_left), channels), dtype=img.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
                     abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = img
        img = expand_image[max(offset_up, 0):max(offset_up, 0) + self.size[1],
                           max(offset_left, 0):max(offset_left, 0) + self.size[0]]
        if maskmap is not None:
            expand_maskmap = np.zeros((max(height, self.size[1]) + abs(offset_up),
                                       max(width, self.size[0]) + abs(offset_left)), dtype=maskmap.dtype)
            expand_maskmap[:, :] = 1
            expand_maskmap[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
                           abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = maskmap
            maskmap = expand_maskmap[max(offset_up, 0):max(offset_up, 0) + self.size[1],
                                     max(offset_left, 0):max(offset_left, 0) + self.size[0]]

        if labelmap is not None:
            expand_labelmap = np.zeros((max(height, self.size[1]) + abs(offset_up),
                                        max(width, self.size[0]) + abs(offset_left)), dtype=labelmap.dtype)
            expand_labelmap[:, :] = 255
            expand_labelmap[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
                            abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = labelmap
            labelmap = expand_labelmap[max(offset_up, 0):max(offset_up, 0) + self.size[1],
                                       max(offset_left, 0):max(offset_left, 0) + self.size[0]]

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


class RandomDetCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    @staticmethod
    def intersect(box_a, box_b):
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]

    @staticmethod
    def jaccard_numpy(box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
            is simply the intersection over union of two boxes.
            E.g.:
                A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
            Args:
                box_a: Multiple bounding boxes, Shape: [num_boxes,4]
                box_b: Single bounding box, Shape: [4]
            Return:
                jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
            """
        inter = RandomDetCrop.intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2] - box_b[0]) *
                  (box_b[3] - box_b[1]))  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None and maskmap is None and kpts is None and polygons is None
        assert bboxes is not None and labels is not None
        if random.random() > self.ratio:
            return img, labelmap, maskmap, kpts, bboxes, labels, polygons

        height, width, _ = img.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None or bboxes.size == 0:
                return img, labelmap, maskmap, kpts, bboxes, labels, polygons

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                scale = random.uniform(0.3, 1.)
                min_ratio = max(0.5, scale * scale)
                max_ratio = min(2.0, 1. / scale / scale)
                ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
                w = int(scale * ratio * width)
                h = int((scale / ratio) * height)
                left = random.randint(0, width - w)
                top = random.randint(0, height - h)
                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = self.jaccard_numpy(bboxes, rect)
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou or max_iou < overlap.max():
                    continue

                # keep overlap with gt box IF center in sampled patch
                centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = bboxes[mask, :].copy()
                # cut the crop from the image
                current_img = img[rect[1]:rect[3], rect[0]:rect[2], :]
                # take only matching gt labels
                current_labels = labels[mask]
                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                return current_img, labelmap, maskmap, kpts, current_boxes, current_labels, polygons


class Resize(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.
    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """
    def __init__(self, target_size=None, min_side_length=None, max_side_length=None):
        self.target_size = target_size
        self.min_side_length = min_side_length
        self.max_side_length = max_side_length

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        assert isinstance(img, (np.ndarray, list))
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)
        height, width, _ = img.shape if isinstance(img, np.ndarray) else img[0].shape
        if self.target_size is not None:
            w_scale_ratio = self.target_size[0] / width
            h_scale_ratio = self.target_size[1] / height

        elif self.min_side_length is not None and self.max_side_length is None:
            scale_ratio = self.min_side_length / min(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio

        elif self.min_side_length is None and self.max_side_length is not None:
            scale_ratio = self.max_side_length / max(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio

        else:
            scale1 = self.min_side_length / min(width, height)
            scale2 = self.max_side_length / max(width, height)
            w_scale_ratio, h_scale_ratio = min(scale1, scale2), min(scale1, scale2)

        target_size = [int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))]

        if kpts is not None and kpts.size > 0:
            kpts[:, :, 0] *= w_scale_ratio
            kpts[:, :, 1] *= h_scale_ratio

        if bboxes is not None and bboxes.size > 0:
            bboxes[:, 0::2] *= w_scale_ratio
            bboxes[:, 1::2] *= h_scale_ratio

        if polygons is not None:
            for object_id in range(len(polygons)):
                for polygon_id in range(len(polygons[object_id])):
                    polygons[object_id][polygon_id][0::2] *= w_scale_ratio
                    polygons[object_id][polygon_id][1::2] *= h_scale_ratio

        target_size = tuple(target_size)
        if not isinstance(img, list):
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            img = [cv2.resize(item, target_size, interpolation=cv2.INTER_LINEAR) for item in img]
        if labelmap is not None:
            labelmap = cv2.resize(labelmap, target_size, interpolation=cv2.INTER_NEAREST)

        if maskmap is not None:
            maskmap = cv2.resize(maskmap, target_size, interpolation=cv2.INTER_NEAREST)

        return img, labelmap, maskmap, kpts, bboxes, labels, polygons


CV2_AUGMENTATIONS_DICT = {
    'random_blur': RandomBlur,
    'random_saturation': RandomSaturation,
    'random_hue': RandomHue,
    'random_perm': RandomPerm,
    'random_contrast': RandomContrast,
    'random_brightness': RandomBrightness,
    'random_erase': RandomErase,
    'random_pad': RandomPad,
    'random_border': RandomBorder,
    'random_hflip': RandomHFlip,
    'random_resize': RandomResize,
    'random_crop': RandomCrop,
    'random_focus_crop': RandomFocusCrop,
    'random_det_crop': RandomDetCrop,
    'random_resized_crop': RandomResizedCrop,
    'random_rotate': RandomRotate,
    'resize': Resize
}


class CV2AugCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> CV2AugCompose([
        >>>     RandomCrop(),
        >>> ])
    """
    def __init__(self, configer, split='train'):
        self.configer = configer
        self.transforms = dict()
        self.split = split
        self.trans_dict = self.configer.get(split, 'aug_trans')
        shuffle_train_trans = []
        if 'shuffle_trans_seq' in self.trans_dict:
            if isinstance(self.trans_dict['shuffle_trans_seq'][0], list):
                train_trans_seq_list = self.trans_dict['shuffle_trans_seq']
                for train_trans_seq in train_trans_seq_list:
                    shuffle_train_trans += train_trans_seq

            else:
                shuffle_train_trans = self.trans_dict['shuffle_trans_seq']

        for trans in self.trans_dict['trans_seq'] + shuffle_train_trans:
            if 'func' in self.trans_dict[trans]:
                self.transforms[trans] = CV2_AUGMENTATIONS_DICT[self.trans_dict[trans]['func']](**self.trans_dict[trans]['params'])
            else:
                self.transforms[trans] = CV2_AUGMENTATIONS_DICT[trans](**self.trans_dict[trans])

    def __call__(self, img, labelmap=None, maskmap=None, kpts=None, bboxes=None, labels=None, polygons=None):
        shuffle_trans_seq = []
        if 'shuffle_trans_seq' in self.trans_dict:
            if isinstance(self.trans_dict['shuffle_trans_seq'][0], list):
                shuffle_trans_seq_list = self.trans_dict['shuffle_trans_seq']
                shuffle_trans_seq = shuffle_trans_seq_list[random.randint(0, len(shuffle_trans_seq_list))]
            else:
                shuffle_trans_seq = self.trans_dict['shuffle_trans_seq']
                random.shuffle(shuffle_trans_seq)

        for trans_key in (shuffle_trans_seq + self.trans_dict['trans_seq']):
            (img, labelmap, maskmap, kpts,
             bboxes, labels, polygons) = self.transforms[trans_key](img, labelmap, maskmap,
                                                                    kpts, bboxes, labels, polygons)

        if self.configer.get('data', 'input_mode') == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.configer.get('data', 'input_mode') == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        out_list = [img]
        for elem in [labelmap, maskmap, kpts, bboxes, labels, polygons]:
            if elem is not None:
                out_list.append(elem)

        return out_list if len(out_list) > 1 else out_list[0]
