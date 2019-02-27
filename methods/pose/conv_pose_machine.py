#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Pose Estimator.


import time
import torch

from datasets.pose.data_loader import DataLoader
from methods.tools.runner_helper import RunnerHelper
from methods.tools.trainer import Trainer
from models.pose.model_manager import ModelManager
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from utils.visualizer.pose_visualizer import PoseVisualizer


class ConvPoseMachine(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
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
        self.pose_net = self.pose_model_manager.single_pose_detector()
        self.pose_net = RunnerHelper.load_net(self, self.pose_net)

        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.configer.get('solver'))

        self.train_loader = self.pose_data_loader.get_trainloader()
        self.val_loader = self.pose_data_loader.get_valloader()

        self.mse_loss = self.pose_model_manager.get_pose_loss()

    def _get_parameters(self):

        return self.pose_net.parameters()

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.pose_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.runner_state['epoch'] += 1

        # data_tuple: (inputs, heatmap, maskmap, tagmap, num_objects)
        for i, data_dict in enumerate(self.train_loader):
            Trainer.update(self, solver_dict=self.configer.get('solver'))
            inputs = data_dict['img']
            heatmap = data_dict['heatmap']

            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs, heatmap = RunnerHelper.to_device(self, inputs, heatmap)
            # self.pose_visualizer.vis_peaks(heatmap[0], inputs[0], name='cpm')

            # Forward pass.
            outputs = self.pose_net(inputs)

            # Compute the loss of the train batch & backward.
            loss = self.mse_loss(outputs, heatmap)

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
        self.pose_net.eval()
        start_time = time.time()

        with torch.no_grad():
            for j, data_dict in enumerate(self.val_loader):
                inputs = data_dict['img']
                heatmap = data_dict['heatmap']
                # Change the data type.
                inputs, heatmap = RunnerHelper.to_device(self, inputs, heatmap)

                # Forward pass.
                outputs = self.pose_net(inputs)

                # Compute the loss of the val batch.
                loss = self.mse_loss(outputs[-1], heatmap)

                self.val_losses.update(loss.item(), inputs.size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            RunnerHelper.save_net(self, self.pose_net, iters=self.runner_state['iters'])
            # Print the log info & reset the states.
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            self.batch_time.reset()
            self.val_losses.reset()
            self.pose_net.train()


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
