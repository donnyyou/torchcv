#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


import time
import torch

from datasets.det.data_loader import DataLoader
from methods.det.single_shot_detector_test import SingleShotDetectorTest
from methods.tools.runner_helper import RunnerHelper
from methods.tools.trainer import Trainer
from models.det.model_manager import ModelManager
from models.det.layers.ssd_priorbox_layer import SSDPriorBoxLayer
from models.det.layers.ssd_target_generator import SSDTargetGenerator
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from metrics.det.det_running_score import DetRunningScore
from utils.visualizer.det_visualizer import DetVisualizer


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
        self.det_model_manager = ModelManager(configer)
        self.det_data_loader = DataLoader(configer)
        self.ssd_target_generator = SSDTargetGenerator(configer)
        self.ssd_priorbox_layer = SSDPriorBoxLayer(configer)
        self.det_running_score = DetRunningScore(configer)

        self.det_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = RunnerHelper.load_net(self, self.det_net)
        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.configer.get('solver'))
        self.train_loader = self.det_data_loader.get_trainloader()
        self.val_loader = self.det_data_loader.get_valloader()
        self.det_loss = self.det_model_manager.get_det_loss()

    def _get_parameters(self):
        lr_1 = []
        lr_10 = []
        params_dict = dict(self.det_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                lr_10.append(value)
            else:
                lr_1.append(value)

        params = [{'params': lr_1, 'lr': self.configer.get('solver', 'lr')['base_lr']},
                  {'params': lr_10, 'lr': self.configer.get('solver', 'lr')['base_lr']}]

        return params

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.det_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.runner_state['epoch'] += 1

        # data_tuple: (inputs, heatmap, maskmap, vecmap)
        for i, data_dict in enumerate(self.train_loader):
            Trainer.update(self, backbone_list=(0,), solver_dict=self.configer.get('solver'))
            inputs = data_dict['img']
            batch_gt_bboxes = data_dict['bboxes']
            batch_gt_labels = data_dict['labels']
            # Change the data type.
            inputs = RunnerHelper.to_device(self, inputs)

            self.data_time.update(time.time() - start_time)
            # Forward pass.
            outputs = self.det_net(inputs)
            if self.configer.get('network', 'gathered'):
                feat_list = outputs[0]
            else:
                feat_list = outputs[0][0]

            bboxes, labels = self.ssd_target_generator(feat_list, batch_gt_bboxes,
                                                       batch_gt_labels, [inputs.size(3), inputs.size(2)])

            bboxes, labels = RunnerHelper.to_device(bboxes, labels)
            # Compute the loss of the train batch & backward.
            loss = self.det_loss(outputs, bboxes, labels, gathered=self.configer.get('network', 'gathered'))

            self.train_losses.update(loss.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.runner_state['iters'] += 1

            # Print the log info & reset the states.
            if self.runner_state['iters'] % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    self.runner_state['epoch'], self.runner_state['iters'],
                    self.configer.get('solver', 'display_iter'),
                    RunnerHelper.get_lr(self.optimizer), batch_time=self.batch_time,
                    data_time=self.data_time, loss=self.train_losses))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            if self.configer.get('solver', 'lr')['metric'] == 'iters' \
                    and self.runner_state['iters'] == self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            if self.runner_state['iters'] % self.configer.get('solver', 'test_interval') == 0:
                self.val()

    def val(self):
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
                inputs = RunnerHelper.to_device(self, inputs)
                input_size = [inputs.size(3), inputs.size(2)]
                # Forward pass.
                outputs = self.det_net(inputs)
                feat_list, loc, cls = RunnerHelper.gather(self, outputs)

                bboxes, labels = self.ssd_target_generator(feat_list, batch_gt_bboxes,
                                                           batch_gt_labels, input_size)

                bboxes, labels = RunnerHelper.to_device(bboxes, labels)
                # Compute the loss of the val batch.
                loss = self.det_loss(outputs, bboxes, labels, gathered=self.configer.get('network', 'gathered'))
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

            RunnerHelper.save_net(self, self.det_net, iters=self.runner_state['iters'])
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


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
