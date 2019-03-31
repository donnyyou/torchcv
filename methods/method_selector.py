#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from methods.cls.image_classifier import ImageClassifier
from methods.cls.image_classifier_test import ImageClassifierTest
from methods.det.faster_rcnn import FasterRCNN
from methods.det.faster_rcnn_test import FastRCNNTest
from methods.det.single_shot_detector import SingleShotDetector
from methods.det.single_shot_detector_test import SingleShotDetectorTest
from methods.det.yolov3 import YOLOv3
from methods.det.yolov3_test import YOLOv3Test
from methods.pose.conv_pose_machine import ConvPoseMachine
from methods.pose.conv_pose_machine_test import ConvPoseMachineTest
from methods.pose.open_pose import OpenPose
from methods.pose.open_pose_test import OpenPoseTest
from methods.seg.fcn_segmentor import FCNSegmentor
from methods.seg.fcn_segmentor_test import FCNSegmentorTest
from methods.gan.image_translator import ImageTranslator
from methods.gan.image_translator_test import ImageTranslatorTest
from methods.gan.face_gan import FaceGAN
from methods.gan.face_gan_test import FaceGANTest
from utils.tools.logger import Logger as Log


POSE_METHOD_DICT = {
    'open_pose': OpenPose,
    'conv_pose_machine': ConvPoseMachine,
}
POSE_TEST_DICT = {
    'open_pose': OpenPoseTest,
    'conv_pose_machine': ConvPoseMachineTest,
}

SEG_METHOD_DICT = {
    'fcn_segmentor': FCNSegmentor,
}
SEG_TEST_DICT = {
    'fcn_segmentor': FCNSegmentorTest,
}

DET_METHOD_DICT = {
    'faster_rcnn': FasterRCNN,
    'single_shot_detector': SingleShotDetector,
    'yolov3': YOLOv3,
}
DET_TEST_DICT = {
    'faster_rcnn': FastRCNNTest,
    'single_shot_detector': SingleShotDetectorTest,
    'yolov3': YOLOv3Test,
}

CLS_METHOD_DICT = {
    'image_classifier': ImageClassifier,
}
CLS_TEST_DICT = {
    'image_classifier': ImageClassifierTest,
}

GAN_METHOD_DICT = {
    'image_translator': ImageTranslator,
    'face_gan': FaceGAN
}
GAN_TEST_DICT = {
    'image_translator': ImageTranslatorTest,
    'face_gan': FaceGANTest
}


class MethodSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def select_pose_method(self):
        key = self.configer.get('method')
        if key not in POSE_METHOD_DICT or key not in POSE_TEST_DICT:
            Log.error('Pose Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return POSE_METHOD_DICT[key](self.configer)
        else:
            return POSE_TEST_DICT[key](self.configer)

    def select_det_method(self):
        key = self.configer.get('method')
        if key not in DET_METHOD_DICT or key not in DET_TEST_DICT:
            Log.error('Det Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return DET_METHOD_DICT[key](self.configer)
        else:
            return DET_TEST_DICT[key](self.configer)

    def select_seg_method(self):
        key = self.configer.get('method')
        if key not in SEG_METHOD_DICT or key not in SEG_TEST_DICT:
            Log.error('Det Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return SEG_METHOD_DICT[key](self.configer)
        else:
            return SEG_TEST_DICT[key](self.configer)

    def select_cls_method(self):
        key = self.configer.get('method')
        if key not in CLS_METHOD_DICT or key not in CLS_TEST_DICT:
            Log.error('Cls Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return CLS_METHOD_DICT[key](self.configer)
        else:
            return CLS_TEST_DICT[key](self.configer)

    def select_gan_method(self):
        key = self.configer.get('method')
        if key not in GAN_METHOD_DICT or key not in GAN_TEST_DICT:
            Log.error('Cls Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return GAN_METHOD_DICT[key](self.configer)
        else:
            return GAN_TEST_DICT[key](self.configer)

