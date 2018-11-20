#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (donnyyou@163.com)
# Class Definition for Single Shot Detector.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.backends.cudnn as cudnn

from datasets.det_data_loader import DetDataLoader
from datasets.tools.data_transformer import DataTransformer
from loss.det_loss_manager import DetLossManager
from methods.det.single_shot_detector_test import SingleShotDetectorTest
from methods.tools.module_utilizer import ModuleUtilizer
from methods.tools.optim_scheduler import OptimScheduler
from models.det_model_manager import DetModelManager
from utils.layers.det.ssd_priorbox_layer import SSDPriorBoxLayer
from utils.layers.det.ssd_target_generator import SSDTargetGenerator
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from val.scripts.det.det_running_score import DetRunningScore
from vis.visualizer.det_visualizer import DetVisualizer


class SingleShotDetector(object):
    """
      The class for Single Shot Detector. Include train, val, test & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.det_visualizer = DetVisualizer(configer)
        self.det_loss_manager = DetLossManager(configer)
        self.det_model_manager = DetModelManager(configer)
        self.det_data_loader = DetDataLoader(configer)
        self.ssd_target_generator = SSDTargetGenerator(configer)
        self.ssd_priorbox_layer = SSDPriorBoxLayer(configer)
        self.det_running_score = DetRunningScore(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_transformer = DataTransformer(configer)

        self.det_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = self.module_utilizer.load_net(self.det_net)

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(self._get_parameters())

        self.train_loader = self.det_data_loader.get_trainloader()
        self.val_loader = self.det_data_loader.get_valloader()

        self.det_loss = self.det_loss_manager.get_det_loss('ssd_multibox_loss')

    def _get_parameters(self):

        return self.det_net.parameters()

    def warm_lr(self, batch_len):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        warm_iters = self.configer.get('lr', 'warm')['warm_epoch'] * batch_len
        warm_lr = self.configer.get('lr', 'warm')['warm_lr']
        if self.configer.get('iters') < warm_iters:
            lr_delta = (self.configer.get('lr', 'base_lr') - warm_lr) * self.configer.get('iters') / warm_iters
            lr = warm_lr + lr_delta

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.det_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.configer.plus_one('epoch')
        self.scheduler.step(self.configer.get('epoch'))

        # data_tuple: (inputs, heatmap, maskmap, vecmap)
        for i, data_dict in enumerate(self.train_loader):
            if not self.configer.is_empty('lr', 'is_warm') and self.configer.get('lr', 'is_warm'):
                self.warm_lr(len(self.train_loader))

            inputs = data_dict['img']
            batch_gt_bboxes = data_dict['bboxes']
            batch_gt_labels = data_dict['labels']
            # Change the data type.
            inputs = self.module_utilizer.to_device(inputs)

            self.data_time.update(time.time() - start_time)
            # Forward pass.
            feat_list, loc, cls = self.det_net(inputs)

            bboxes, labels = self.ssd_target_generator(feat_list, batch_gt_bboxes,
                                                       batch_gt_labels, [inputs.size(3), inputs.size(2)])

            bboxes, labels = self.module_utilizer.to_device(bboxes, labels)
            # Compute the loss of the train batch & backward.
            loss = self.det_loss(loc, bboxes, cls, labels)

            self.train_losses.update(loss.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    self.configer.get('epoch'), self.configer.get('iters'),
                    self.configer.get('solver', 'display_iter'),
                    self.scheduler.get_lr(), batch_time=self.batch_time,
                    data_time=self.data_time, loss=self.train_losses))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # Check to val the current model.
            if self.val_loader is not None and \
               (self.configer.get('iters')) % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

    def __val(self):
        """
          Validation function during the train phase.
        """
        self.det_net.eval()
        start_time = time.time()
        with torch.no_grad():
            for j, data_dict in enumerate(self.val_loader):
                inputs = data_dict['img']
                batch_gt_bboxes = data_dict['bboxes']
                batch_gt_labels = data_dict['labels']
                inputs = self.module_utilizer.to_device(inputs)
                input_size = [inputs.size(3), inputs.size(2)]
                # Forward pass.
                feat_list, loc, cls = self.det_net(inputs)
                bboxes, labels = self.ssd_target_generator(feat_list, batch_gt_bboxes,
                                                           batch_gt_labels, input_size)

                bboxes, labels = self.module_utilizer.to_device(bboxes, labels)
                # Compute the loss of the val batch.
                loss = self.det_loss(loc, bboxes, cls, labels)
                self.val_losses.update(loss.item(), inputs.size(0))

                batch_detections = SingleShotDetectorTest.decode(loc, cls,
                                                                 self.ssd_priorbox_layer(feat_list, input_size),
                                                                 self.configer, input_size)
                batch_pred_bboxes = self.__get_object_list(batch_detections)
                # batch_pred_bboxes = self._get_gt_object_list(batch_gt_bboxes, batch_gt_labels)
                self.det_running_score.update(batch_pred_bboxes, batch_gt_bboxes, batch_gt_labels)

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.module_utilizer.save_net(self.det_net, save_mode='iters')
            # Print the log info & reset the states.
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            Log.info('Val mAP: {}'.format(self.det_running_score.get_mAP()))
            self.det_running_score.reset()
            self.batch_time.reset()
            self.val_losses.reset()
            self.det_net.train()

    def _get_gt_object_list(self, batch_gt_bboxes, batch_gt_labels):
        batch_pred_bboxes = list()
        for i in range(len(batch_gt_bboxes)):
            object_list = list()
            if batch_gt_bboxes[i].numel() > 0:
                for j in range(batch_gt_bboxes[i].size(0)):
                    object_list.append([batch_gt_bboxes[i][j][0].item(), batch_gt_bboxes[i][j][1].item(),
                                        batch_gt_bboxes[i][j][2].item(), batch_gt_bboxes[i][j][3].item(),
                                        batch_gt_labels[i][j].item(), 1.0])

            batch_pred_bboxes.append(object_list)
        return batch_pred_bboxes

    def __get_object_list(self, batch_detections):
        batch_pred_bboxes = list()
        for idx, detections in enumerate(batch_detections):
            object_list = list()
            if detections is not None:
                for x1, y1, x2, y2, conf, cls_pred in detections:
                    xmin = x1.cpu().item()
                    ymin = y1.cpu().item()
                    xmax = x2.cpu().item()
                    ymax = y2.cpu().item()
                    cf = conf.cpu().item()
                    cls_pred = cls_pred.cpu().item() - 1
                    object_list.append([xmin, ymin, xmax, ymax, int(cls_pred), float('%.2f' % cf)])

            batch_pred_bboxes.append(object_list)

        return batch_pred_bboxes

    def train(self):
        cudnn.benchmark = True
        if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
            self.__val()

        while self.configer.get('epoch') < self.configer.get('solver', 'max_epoch'):
            self.__train()
            if self.configer.get('epoch') == self.configer.get('solver', 'max_epoch'):
                break


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
