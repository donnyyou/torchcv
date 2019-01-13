#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Repackage some image operations.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from PIL import Image

from utils.tools.logger import Logger as Log


PIL_INTER_DICT = {
    'nearest': Image.NEAREST,
    'linear': Image.BILINEAR,
    'cubic': Image.CUBIC
}

CV2_INTER_DICT = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC
}


class ImageHelper(object):

    @staticmethod
    def read_image(image_path, tool='pil', mode='RGB'):
        if tool == 'pil':
            return ImageHelper.pil_read_image(image_path, mode=mode)
        elif tool == 'cv2':
            return ImageHelper.cv2_read_image(image_path, mode=mode)
        else:
            Log.error('Not support mode {}'.format(mode))
            exit(1)

    @staticmethod
    def cv2_read_image(image_path, mode='RGB'):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mode == 'RGB':
            return ImageHelper.bgr2rgb(img_bgr)

        elif mode == 'BGR':
            return img_bgr

        elif mode == 'P':
            return ImageHelper.img2np(Image.open(image_path).convert('P'))

        else:
            Log.error('Not support mode {}'.format(mode))
            exit(1)

    @staticmethod
    def pil_read_image(image_path, mode='RGB'):
        with open(image_path, 'rb') as f:
            img = Image.open(f)

            if mode == 'RGB':
                return img.convert('RGB')

            elif mode == 'BGR':
                img = img.convert('RGB')
                cv_img = ImageHelper.rgb2bgr(np.array(img))
                return Image.fromarray(cv_img)

            elif mode == 'P':
                return img.convert('P')

            else:
                Log.error('Not support mode {}'.format(mode))
                exit(1)

    @staticmethod
    def rgb2bgr(img_rgb):
        if isinstance(img_rgb, Image.Image):
            img_bgr = ImageHelper.rgb2bgr(ImageHelper.img2np(img_rgb))
            return ImageHelper.np2img(img_bgr)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr

    @staticmethod
    def bgr2rgb(img_bgr):
        if isinstance(img_bgr, Image.Image):
            img_rgb = ImageHelper.bgr2rgb(ImageHelper.img2np(img_bgr))
            return ImageHelper.np2img(img_rgb)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    @staticmethod
    def bgr2gray(img, keepdim=False):
        """Convert a BGR image to grayscale image.

        Args:
            img (ndarray): The input image.
            keepdim (bool): If False (by default), then return the grayscale image
                with 2 dims, otherwise 3 dims.

        Returns:
            ndarray: The converted grayscale image.
        """
        out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if keepdim:
            out_img = out_img[..., None]
        return out_img

    @staticmethod
    def gray2bgr(img):
        """Convert a grayscale image to BGR image.

        Args:
            img (ndarray or str): The input image.

        Returns:
            ndarray: The converted BGR image.
        """
        img = img[..., None] if img.ndim == 2 else img
        out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return out_img

    @staticmethod
    def get_cv2_bgr(img, mode='RGB'):
        if isinstance(img, Image.Image):
            img = ImageHelper.img2np(img)

        if mode == 'RGB':
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img_bgr

        return img

    @staticmethod
    def imshow(win_name, img, time=0):
        if isinstance(img, Image.Image):
            img = ImageHelper.rgb2bgr(ImageHelper.img2np(img))

        cv2.imshow(win_name, img)
        cv2.waitKey(time)

    @staticmethod
    def np2img(arr):
        if len(arr.shape) == 2:
            mode = 'P'
        else:
            mode = 'RGB'

        return Image.fromarray(arr, mode=mode)

    @staticmethod
    def img2np(img):
        return np.array(img)

    @staticmethod
    def tonp(img):
        if isinstance(img, Image.Image):
            img = ImageHelper.img2np(img)

        return img.astype(np.uint8)

    @staticmethod
    def get_size(img):
        if isinstance(img, Image.Image):
            return img.size

        elif isinstance(img, np.ndarray):
            height, width = img.shape[:2]
            return [width, height]

        else:
            Log.error('Image type is invalid.')
            exit(1)

    @staticmethod
    def resize(img, target_size, interpolation=None):
        assert isinstance(target_size, (list, tuple))
        assert isinstance(interpolation, str)

        target_size = tuple(target_size)
        if isinstance(img, Image.Image):
            return ImageHelper.pil_resize(img, target_size, interpolation=PIL_INTER_DICT[interpolation])

        elif isinstance(img, np.ndarray):
            return ImageHelper.cv2_resize(img, target_size, interpolation=CV2_INTER_DICT[interpolation])

        else:
            Log.error('Image type is invalid.')
            exit(1)

    @staticmethod
    def pil_resize(img, target_size, interpolation):
        assert isinstance(target_size, (list, tuple))

        target_size = tuple(target_size)

        if isinstance(img, Image.Image):
            return img.resize(target_size, interpolation)

        elif isinstance(img, np.ndarray):
            pil_img = ImageHelper.np2img(img)
            return ImageHelper.img2np(pil_img.resize(target_size, interpolation))

        else:
            Log.error('Image type is invalid.')
            exit(1)

    @staticmethod
    def cv2_resize(img, target_size, interpolation):
        assert isinstance(target_size, (list, tuple))

        target_size = tuple(target_size)

        if isinstance(img, Image.Image):
            img = ImageHelper.img2np(img)
            target_img = cv2.resize(img, target_size, interpolation=interpolation)
            return ImageHelper.np2img(target_img)

        elif isinstance(img, np.ndarray):
            return cv2.resize(img, target_size, interpolation=interpolation)

        else:
            Log.error('Image type is invalid.')
            exit(1)

    @staticmethod
    def save(img, save_path):
        if isinstance(img, Image.Image):
            img.save(save_path)

        elif isinstance(img, np.ndarray):
            cv2.imwrite(save_path, img)

        else:
            Log.error('Image type is invalid.')
            exit(1)

    @staticmethod
    def fig2img(fig):
        """
        @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
        @param fig a matplotlib figure
        @return a Python Imaging Library ( PIL ) image
        """
        # put the figure pixmap into a numpy array
        buf = ImageHelper.fig2data(fig)
        h, w, d = buf.shape
        return Image.frombytes("RGBA", (w, h), buf.tostring())

    @staticmethod
    def fig2np(fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    @staticmethod
    def fig2data(fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        return buf.reshape(h, w, 4)

    @staticmethod
    def imfrombytes(content, flag='color'):
        """Read an image from bytes.

        Args:
            content (bytes): Image bytes got from files or other streams.
            flag (str): Same as :func:`imread`.

        Returns:
            ndarray: Loaded image array.
        """
        imread_flags = {
            'color': cv2.IMREAD_COLOR,
            'grayscale': cv2.IMREAD_GRAYSCALE,
            'unchanged': cv2.IMREAD_UNCHANGED
        }
        img_np = np.fromstring(content, np.uint8)
        flag = imread_flags[flag] if isinstance(flag, str) else flag
        img = cv2.imdecode(img_np, flag)
        return img

    @staticmethod
    def is_img(img_name):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]
        return any(img_name.endswith(extension) for extension in IMG_EXTENSIONS)


if __name__ == "__main__":
    target_size = (368, 368)
    image_path = '/home/donny/Projects/PyTorchCV/val/samples/pose/coco/ski.jpg'
    pil_img = ImageHelper.cv2_read_image(image_path)
    pil_img = ImageHelper.np2img(pil_img)
    cv2_img = ImageHelper.cv2_read_image(image_path)
    ImageHelper.imshow('main', np.array(pil_img) - cv2_img)

    pil_img = ImageHelper.cv2_resize(pil_img, target_size, interpolation=cv2.INTER_CUBIC)
    cv2_img = ImageHelper.cv2_resize(cv2_img, target_size, interpolation=cv2.INTER_CUBIC)
    # cv2_img = ImageHelper.bgr2rgb(cv2_img)
    ImageHelper.imshow('main', np.array(pil_img) - cv2_img)
    ImageHelper.imshow('main', pil_img)
    ImageHelper.imshow('main', cv2_img)

    # resize_pil_img.show()
    print(np.unique(np.array(pil_img) - np.array(cv2_img)))
