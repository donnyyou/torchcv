#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch

from datasets.seg_data_loader import SegDataLoader
from loss.loss_manager import LossManager
from methods.tools.runner_helper import RunnerHelper
from methods.tools.trainer import Trainer
from models.seg_model_manager import SegModelManager
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from val.scripts.seg.seg_running_score import SegRunningScore
from vis.visualizer.seg_visualizer import SegVisualizer


class FCNSegmentor(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_running_score = SegRunningScore(configer)
        self.seg_visualizer = SegVisualizer(configer)
        self.seg_loss_manager = LossManager(configer)
        self.seg_model_manager = SegModelManager(configer)
        self.seg_data_loader = SegDataLoader(configer)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        self.seg_net = self.seg_model_manager.semantic_segmentor()
        self.seg_net = RunnerHelper.load_net(self, self.seg_net)

        self.optimizer, self.scheduler = Trainer.init(self, self._get_parameters())

        self.train_loader = self.seg_data_loader.get_trainloader()
        self.val_loader = self.seg_data_loader.get_valloader()

        self.pixel_loss = self.seg_loss_manager.get_seg_loss()

    def _get_parameters(self):
        lr_1 = []
        lr_10 = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                lr_10.append(value)
            else:
                lr_1.append(value)

        params = [{'params': lr_1, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': lr_10, 'lr': self.configer.get('lr', 'base_lr') * 1.0}]
        return params

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.

        for i, data_dict in enumerate(self.train_loader):
            Trainer.update(self, backbone_list=(0, ))
            inputs = data_dict['img']
            targets = data_dict['labelmap']
            self.data_time.update(time.time() - start_time)
            # Change the data type.

            inputs, targets = RunnerHelper.to_device(self, inputs, targets)

            # Forward pass.
            outputs = self.seg_net(inputs)
            # outputs = self.module_utilizer.gather(outputs)
            # Compute the loss of the train batch & backward.
            loss = self.pixel_loss(outputs, targets, gathered=self.configer.get('network', 'gathered'))
            self.train_losses.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.runner_state['iters'] += 1

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:
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

            if self.configer.get('lr', 'metric') == 'iters' \
                    and self.runner_state['iters'] == self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            if self.runner_state['iters'] % self.configer.get('solver', 'test_interval') == 0:
                self.val()

        self.runner_state['epoch'] += 1

    def val(self):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        start_time = time.time()

        for j, data_dict in enumerate(self.val_loader):
            inputs = data_dict['img']
            targets = data_dict['labelmap']

            with torch.no_grad():
                # Change the data type.
                inputs, targets = RunnerHelper.to_device(self, inputs, targets)
                # Forward pass.
                outputs = self.seg_net(inputs)
                # Compute the loss of the val batch.
                loss = self.pixel_loss(outputs, targets, gathered=self.configer.get('network', 'gathered'))
                outputs = RunnerHelper.gather(self, outputs)
                pred = outputs[0]

            self.val_losses.update(loss.item(), inputs.size(0))
            self.seg_running_score.update(pred.max(1)[1].cpu().numpy(), targets.cpu().numpy())

            # Update the vars of the val phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.runner_state['performance'] = self.seg_running_score.get_mean_iou()
        self.runner_state['val_loss'] = self.val_losses.avg
        RunnerHelper.save_net(self, self.seg_net,
                              performance=self.seg_running_score.get_mean_iou(),
                              val_loss=self.val_losses.avg)

        # Print the log info & reset the states.
        Log.info(
            'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
            'Loss {loss.avg:.8f}\n'.format(
                batch_time=self.batch_time, loss=self.val_losses))
        Log.info('Mean IOU: {}\n'.format(self.seg_running_score.get_mean_iou()))
        Log.info('Pixel ACC: {}\n'.format(self.seg_running_score.get_pixel_acc()))
        self.batch_time.reset()
        self.val_losses.reset()
        self.seg_running_score.reset()
        self.seg_net.train()


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
