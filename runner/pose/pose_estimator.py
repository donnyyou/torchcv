#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Pose Estimator.


import time
import torch

from data.pose.data_loader import DataLoader
from lib.runner.runner_helper import RunnerHelper
from lib.runner.trainer import Trainer
from model.pose.model_manager import ModelManager
from lib.tools.helper.dc_helper import DCHelper
from lib.tools.util.average_meter import AverageMeter, DictAverageMeter
from lib.tools.util.logger import Logger as Log
from lib.tools.vis.pose_visualizer import PoseVisualizer


class PoseEstimator(object):
    """
      The class for Pose Estimation. Include train, val, test & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = DictAverageMeter()
        self.val_losses = DictAverageMeter()
        self.pose_visualizer = PoseVisualizer(configer)
        self.pose_model_manager = ModelManager(configer)
        self.pose_data_loader = DataLoader(configer)

        self.pose_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        self.pose_net = self.pose_model_manager.get_pose_model()
        self.pose_net = RunnerHelper.load_net(self, self.pose_net)

        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.configer.get('solver'))

        self.train_loader = self.pose_data_loader.get_trainloader()
        self.val_loader = self.pose_data_loader.get_valloader()

        self.pose_loss = self.pose_model_manager.get_pose_loss()

    def _get_parameters(self):
        lr_1 = []
        lr_2 = []
        params_dict = dict(self.pose_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                lr_2.append(value)
            else:
                lr_1.append(value)

        params = [{'params': lr_1, 'lr': self.configer.get('solver', 'lr')['base_lr'], 'weight_decay': 0.0},
                  {'params': lr_2, 'lr': self.configer.get('solver', 'lr')['base_lr'], 'weight_decay': 0.0},]

        return params

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.pose_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.runner_state['epoch'] += 1
        for i, data_dict in enumerate(self.train_loader):
            Trainer.update(self, warm_list=(0,), solver_dict=self.configer.get('solver'))
            self.data_time.update(time.time() - start_time)
            # Forward pass.
            out = self.pose_net(data_dict)

            # Compute the loss of the train batch & backward.
            loss_dict = self.pose_loss(out)

            loss = loss_dict['loss']
            self.train_losses.update({key: loss.item() for key, loss in loss_dict.items()}, data_dict['img'].size(0))

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
                         'Learning rate = {4}\tLoss = {3}\n'.format(
                    self.runner_state['epoch'], self.runner_state['iters'],
                    self.configer.get('solver', 'display_iter'), self.train_losses.info(),
                    RunnerHelper.get_lr(self.optimizer), batch_time=self.batch_time, data_time=self.data_time))

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
        self.pose_net.eval()
        start_time = time.time()

        with torch.no_grad():
            for i, data_dict in enumerate(self.val_loader):
                # Forward pass.
                out = self.pose_net(data_dict)
                # Compute the loss of the val batch.
                loss_dict = self.pose_loss(out)

                self.val_losses.update({key: loss.item() for key, loss in loss_dict.items()}, data_dict['img'].size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.runner_state['val_loss'] = self.val_losses.avg['loss']
            RunnerHelper.save_net(self, self.pose_net, val_loss=self.val_losses.avg['loss'])
            # Print the log info & reset the states.
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {0}\n'.format(self.val_losses.info(), batch_time=self.batch_time))
            self.batch_time.reset()
            self.val_losses.reset()
            self.pose_net.train()


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
