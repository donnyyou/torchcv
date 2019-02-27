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


class OpenPose(object):
    """
      The class for Pose Estimation. Include train, val, test & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_schedule_loss = AverageMeter()
        self.train_losses = AverageMeter()
        self.train_loss_heatmap = AverageMeter()
        self.train_loss_associate = AverageMeter()
        self.val_losses = AverageMeter()
        self.val_loss_heatmap = AverageMeter()
        self.val_loss_associate = AverageMeter()
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
        self.pose_net = self.pose_model_manager.multi_pose_detector()
        self.pose_net = RunnerHelper.load_net(self, self.pose_net)

        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.configer.get('solver'))

        self.train_loader = self.pose_data_loader.get_trainloader()
        self.val_loader = self.pose_data_loader.get_valloader()

        self.weights = self.configer.get('network', 'loss_weights')
        self.mse_loss = self.pose_model_manager.get_pose_loss()

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
        self.scheduler.step(self.train_schedule_loss.avg, epoch=self.configer.get('epoch'))
        self.train_schedule_loss.reset()
        # data_tuple: (inputs, heatmap, maskmap, vecmap)
        for i, data_dict in enumerate(self.train_loader):
            inputs = data_dict['img']
            maskmap = data_dict['maskmap']
            heatmap = data_dict['heatmap']
            vecmap = data_dict['vecmap']

            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs, heatmap, maskmap, vecmap = RunnerHelper.to_device(self, inputs, heatmap, maskmap, vecmap)

            # Forward pass.
            paf_out, heatmap_out = self.pose_net(inputs)

            # Compute the loss of the train batch & backward.
            loss_heatmap = self.mse_loss(heatmap_out, heatmap, mask=maskmap, weights=self.weights)
            loss_associate = self.mse_loss(paf_out, vecmap, mask=maskmap, weights=self.weights)
            loss = 2.0 * loss_heatmap + loss_associate

            self.train_losses.update(loss.item(), inputs.size(0))
            self.train_schedule_loss.update(loss.item(), inputs.size(0))
            self.train_loss_heatmap.update(loss_heatmap.item(), inputs.size(0))
            self.train_loss_associate.update(loss_associate.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.runner_state['iters'] += 1

            # Print the log info & reset the states.
            if self.runner_state['iters'] % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Loss Heatmap:{}, Loss Asso: {}'.format(self.train_loss_heatmap.avg,
                                                                 self.train_loss_associate.avg))
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
                self.train_loss_heatmap.reset()
                self.train_loss_associate.reset()

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
                inputs = data_dict['img']
                maskmap = data_dict['maskmap']
                heatmap = data_dict['heatmap']
                vecmap = data_dict['vecmap']
                # Change the data type.
                inputs, heatmap, maskmap, vecmap = RunnerHelper.to_device(self, inputs, heatmap, maskmap, vecmap)

                # Forward pass.
                paf_out, heatmap_out = self.pose_net(inputs)
                # Compute the loss of the val batch.
                loss_heatmap = self.mse_loss(heatmap_out[-1], heatmap, maskmap)
                loss_associate = self.mse_loss(paf_out[-1], vecmap, maskmap)
                loss = 2.0 * loss_heatmap + loss_associate

                self.val_losses.update(loss.item(), inputs.size(0))
                self.val_loss_heatmap.update(loss_heatmap.item(), inputs.size(0))
                self.val_loss_associate.update(loss_associate.item(), inputs.size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.runner_state['val_loss'] = self.val_losses.avg
            RunnerHelper.save_net(self, self.pose_net, val_loss=self.val_losses.avg)
            Log.info('Loss Heatmap:{}, Loss Asso: {}'.format(self.val_loss_heatmap.avg, self.val_loss_associate.avg))
            # Print the log info & reset the states.
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            self.batch_time.reset()
            self.val_losses.reset()
            self.val_loss_heatmap.reset()
            self.val_loss_associate.reset()
            self.pose_net.train()


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
