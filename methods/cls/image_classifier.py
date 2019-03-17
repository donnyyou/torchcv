#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Image Classifier.


import time
import torch

from datasets.cls.data_loader import DataLoader
from methods.tools.runner_helper import RunnerHelper
from methods.tools.trainer import Trainer
from models.cls.model_manager import ModelManager
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from metrics.cls.cls_running_score import ClsRunningScore


class ImageClassifier(object):
    """
      The class for the training phase of Image classification.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.cls_model_manager = ModelManager(configer)
        self.cls_data_loader = DataLoader(configer)
        self.cls_running_score = ClsRunningScore(configer)

        self.cls_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        self.cls_net = self.cls_model_manager.image_classifier()
        self.cls_net = RunnerHelper.load_net(self, self.cls_net)
        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.configer.get('solver'))

        self.train_loader = self.cls_data_loader.get_trainloader()
        self.val_loader = self.cls_data_loader.get_valloader()

        self.ce_loss = self.cls_model_manager.get_cls_loss()

    def _get_parameters(self):

        return self.cls_net.parameters()

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.cls_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.runner_state['epoch'] += 1

        for i, data_dict in enumerate(self.train_loader):
            Trainer.update(self, solver_dict=self.configer.get('solver'))
            inputs = data_dict['img']
            labels = data_dict['label']
            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs, labels = RunnerHelper.to_device(self, inputs, labels)
            # Forward pass.
            outputs = self.cls_net(inputs)
            # Compute the loss of the train batch & backward.

            loss = self.ce_loss(outputs, labels, gathered=self.configer.get('network', 'gathered'))

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
        self.cls_net.eval()
        start_time = time.time()

        with torch.no_grad():
            for j, data_dict in enumerate(self.val_loader):
                inputs = data_dict['img']
                labels = data_dict['label']
                # Change the data type.
                inputs, labels = RunnerHelper.to_device(self, inputs, labels)
                # Forward pass.
                outputs = self.cls_net(inputs)
                # Compute the loss of the val batch.
                loss = self.ce_loss(outputs, labels, gathered=self.configer.get('network', 'gathered'))
                outputs = RunnerHelper.gather(self, outputs)
                self.cls_running_score.update(outputs, labels)
                self.val_losses.update(loss.item(), inputs.size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            RunnerHelper.save_net(self, self.cls_net, performance=self.cls_running_score.get_top1_acc())
            self.runner_state['performance'] = self.cls_running_score.get_top1_acc()
            # Print the log info & reset the states.
            Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))
            Log.info('TestLoss = {loss.avg:.8f}'.format(loss=self.val_losses))
            Log.info('Top1 ACC = {}'.format(self.cls_running_score.get_top1_acc()))
            Log.info('Top5 ACC = {}'.format(self.cls_running_score.get_top5_acc()))
            self.batch_time.reset()
            self.val_losses.reset()
            self.cls_running_score.reset()
            self.cls_net.train()


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
