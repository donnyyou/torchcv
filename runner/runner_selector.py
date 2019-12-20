#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from runner.cls.image_classifier import ImageClassifier
from runner.cls.image_classifier_test import ImageClassifierTest
from runner.det.faster_rcnn import FasterRCNN
from runner.det.faster_rcnn_test import FastRCNNTest
from runner.det.single_shot_detector import SingleShotDetector
from runner.det.single_shot_detector_test import SingleShotDetectorTest
from runner.det.yolov3 import YOLOv3
from runner.det.yolov3_test import YOLOv3Test
from runner.pose.pose_estimator import PoseEstimator
from runner.pose.conv_pose_machine_test import ConvPoseMachineTest
from runner.pose.open_pose_test import OpenPoseTest
from runner.seg.fcn_segmentor import FCNSegmentor
from runner.seg.fcn_segmentor_test import FCNSegmentorTest
from runner.gan.image_translator import ImageTranslator
from runner.gan.image_translator_test import ImageTranslatorTest
from runner.gan.face_gan import FaceGAN
from runner.gan.face_gan_test import FaceGANTest
from lib.tools.util.logger import Logger as Log


POSE_METHOD_DICT = {
    'open_pose': PoseEstimator,
    'conv_pose_machine': PoseEstimator,
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


class RunnerSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def pose_runner(self):
        key = self.configer.get('method')
        if key not in POSE_METHOD_DICT or key not in POSE_TEST_DICT:
            Log.error('Pose Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return POSE_METHOD_DICT[key](self.configer)
        else:
            return POSE_TEST_DICT[key](self.configer)

    def det_runner(self):
        key = self.configer.get('method')
        if key not in DET_METHOD_DICT or key not in DET_TEST_DICT:
            Log.error('Det Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return DET_METHOD_DICT[key](self.configer)
        else:
            return DET_TEST_DICT[key](self.configer)

    def seg_runner(self):
        key = self.configer.get('method')
        if key not in SEG_METHOD_DICT or key not in SEG_TEST_DICT:
            Log.error('Det Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return SEG_METHOD_DICT[key](self.configer)
        else:
            return SEG_TEST_DICT[key](self.configer)

    def cls_runner(self):
        key = self.configer.get('method')
        if key not in CLS_METHOD_DICT or key not in CLS_TEST_DICT:
            Log.error('Cls Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return CLS_METHOD_DICT[key](self.configer)
        else:
            return CLS_TEST_DICT[key](self.configer)

    def gan_runner(self):
        key = self.configer.get('method')
        if key not in GAN_METHOD_DICT or key not in GAN_TEST_DICT:
            Log.error('Cls Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return GAN_METHOD_DICT[key](self.configer)
        else:
            return GAN_TEST_DICT[key](self.configer)

