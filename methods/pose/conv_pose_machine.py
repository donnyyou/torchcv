#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (donnyyou@163.com)
# Class Definition for Pose Estimator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.backends.cudnn as cudnn

from datasets.pose_data_loader import PoseDataLoader
from datasets.tools.data_transformer import DataTransformer
from loss.pose_loss_manager import PoseLossManager
from methods.tools.module_utilizer import ModuleUtilizer
from methods.tools.optim_scheduler import OptimScheduler
from models.pose_model_manager import PoseModelManager
from utils.layers.pose.heatmap_generator import HeatmapGenerator
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from vis.visualizer.pose_visualizer import PoseVisualizer


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
        self.pose_loss_manager = PoseLossManager(configer)
        self.pose_model_manager = PoseModelManager(configer)
        self.pose_data_loader = PoseDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_transformer = DataTransformer(configer)
        self.heatmap_generator = HeatmapGenerator(configer)

        self.pose_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        self._init_model()

    def _init_model(self):
        self.pose_net = self.pose_model_manager.single_pose_detector()
        self.pose_net = self.module_utilizer.load_net(self.pose_net)

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(self._get_parameters())

        self.train_loader = self.pose_data_loader.get_trainloader()
        self.val_loader = self.pose_data_loader.get_valloader()

        self.mse_loss = self.pose_loss_manager.get_pose_loss('mse_loss')

    def _get_parameters(self):

        return self.pose_net.parameters()

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.pose_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.configer.plus_one('epoch')
        self.scheduler.step(self.configer.get('epoch'))

        # data_tuple: (inputs, heatmap, maskmap, tagmap, num_objects)
        for i, data_dict in enumerate(self.train_loader):
            inputs = data_dict['img']
            input_size = [inputs.size(3), inputs.size(2)]
            heatmap = self.heatmap_generator(data_dict['kpts'], input_size)

            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs, heatmap = self.module_utilizer.to_device(inputs, heatmap)
            # self.pose_visualizer.vis_peaks(heatmap[0], inputs[0], name='cpm')

            # Forward pass.
            outputs = self.pose_net(inputs)

            # Compute the loss of the train batch & backward.
            loss = self.mse_loss(outputs, heatmap, maskmap)

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
        self.pose_net.eval()
        start_time = time.time()

        with torch.no_grad():
            for j, data_dict in enumerate(self.val_loader):
                inputs = data_dict['img']
                input_size = [inputs.size(3), inputs.size(2)]
                heatmap = self.heatmap_generator(data_dict['kpts'], input_size)
                # Change the data type.
                inputs, heatmap = self.module_utilizer.to_device(inputs, heatmap)

                # Forward pass.
                outputs = self.pose_net(inputs)

                # Compute the loss of the val batch.
                loss = self.mse_loss(outputs[-1], heatmap)

                self.val_losses.update(loss.item(), inputs.size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.module_utilizer.save_net(self.pose_net, save_mode='iters')
            # Print the log info & reset the states.
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            self.batch_time.reset()
            self.val_losses.reset()
            self.pose_net.train()

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
