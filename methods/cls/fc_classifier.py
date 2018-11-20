#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (donnyyou@163.com)
# Class Definition for Image Classifier.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.backends.cudnn as cudnn

from datasets.cls_data_loader import ClsDataLoader
from loss.cls_loss_manager import ClsLossManager
from methods.tools.module_utilizer import ModuleUtilizer
from methods.tools.optim_scheduler import OptimScheduler
from models.cls_model_manager import ClsModelManager
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from val.scripts.cls.cls_running_score import ClsRunningScore


class FCClassifier(object):
    """
      The class for the training phase of Image classification.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.cls_loss_manager = ClsLossManager(configer)
        self.cls_model_manager = ClsModelManager(configer)
        self.cls_data_loader = ClsDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.cls_running_score = ClsRunningScore(configer)

        self.cls_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        self._init_model()

    def _init_model(self):
        self.cls_net = self.cls_model_manager.image_classifier()
        self.cls_net = self.module_utilizer.load_net(self.cls_net)
        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(self._get_parameters())

        self.train_loader = self.cls_data_loader.get_trainloader()
        self.val_loader = self.cls_data_loader.get_valloader()

        self.ce_loss = self.cls_loss_manager.get_cls_loss('cross_entropy_loss')

    def _get_parameters(self):

        return self.cls_net.parameters()

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.cls_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.configer.plus_one('epoch')
        self.scheduler.step(self.configer.get('epoch'))

        for i, data_dict in enumerate(self.train_loader):
            inputs = data_dict['img']
            labels = data_dict['label']
            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs, labels = self.module_utilizer.to_device(inputs, labels)
            # Forward pass.
            outputs = self.cls_net(inputs)
            outputs = self.module_utilizer.gather(outputs)
            # Compute the loss of the train batch & backward.

            loss = self.ce_loss(outputs, labels)

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
               self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

    def __val(self):
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
                inputs, labels = self.module_utilizer.to_device(inputs, labels)
                # Forward pass.
                outputs = self.cls_net(inputs)
                outputs = self.module_utilizer.gather(outputs)
                # Compute the loss of the val batch.
                loss = self.ce_loss(outputs, labels)
                self.cls_running_score.update(outputs, labels)
                self.val_losses.update(loss.item(), inputs.size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.module_utilizer.save_net(self.cls_net, save_mode='iters')

            # Print the log info & reset the states.
            Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))
            Log.info('TestLoss = {loss.avg:.8f}'.format(loss=self.val_losses))
            Log.info('Top1 ACC = {}'.format(self.cls_running_score.get_top1_acc()))
            Log.info('Top5 ACC = {}'.format(self.cls_running_score.get_top5_acc()))
            self.batch_time.reset()
            self.val_losses.reset()
            self.cls_running_score.reset()
            self.cls_net.train()

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
