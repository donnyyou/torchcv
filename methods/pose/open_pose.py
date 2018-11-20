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
from utils.layers.pose.paf_generator import PafGenerator
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from vis.visualizer.pose_visualizer import PoseVisualizer


class OpenPose(object):
    """
      The class for Pose Estimation. Include train, val, test & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.train_loss_heatmap = AverageMeter()
        self.train_loss_associate = AverageMeter()
        self.val_losses = AverageMeter()
        self.val_loss_heatmap = AverageMeter()
        self.val_loss_associate = AverageMeter()
        self.pose_visualizer = PoseVisualizer(configer)
        self.pose_loss_manager = PoseLossManager(configer)
        self.pose_model_manager = PoseModelManager(configer)
        self.pose_data_loader = PoseDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.heatmap_generator = HeatmapGenerator(configer)
        self.paf_generator = PafGenerator(configer)
        self.data_transformer = DataTransformer(configer)

        self.pose_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        self._init_model()

    def _init_model(self):
        self.pose_net = self.pose_model_manager.multi_pose_detector()
        self.pose_net = self.module_utilizer.load_net(self.pose_net)

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(self._get_parameters())

        self.train_loader = self.pose_data_loader.get_trainloader()
        self.val_loader = self.pose_data_loader.get_valloader()

        self.weights = self.configer.get('network', 'loss_weights')
        self.mse_loss = self.pose_loss_manager.get_pose_loss('mse_loss')

    def _get_parameters(self):
        lr_1 = []
        lr_2 = []
        lr_4 = []
        lr_8 = []
        params_dict = dict(self.pose_net.named_parameters())
        for key, value in params_dict.items():
            if ('model1_' not in key) and ('model0.' not in key) and ('backbone.' not in key):
                if key[-4:] == 'bias':
                    lr_8.append(value)
                else:
                    lr_4.append(value)
            elif key[-4:] == 'bias':
                lr_2.append(value)
            else:
                lr_1.append(value)

        params = [{'params': lr_1, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': lr_2, 'lr': self.configer.get('lr', 'base_lr') * 2., 'weight_decay': 0.0},
                  {'params': lr_4, 'lr': self.configer.get('lr', 'base_lr') * 4.},
                  {'params': lr_8, 'lr': self.configer.get('lr', 'base_lr') * 8., 'weight_decay': 0.0}]

        return params

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.pose_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.configer.plus_one('epoch')
        self.scheduler.step(self.configer.get('epoch'))

        # data_tuple: (inputs, heatmap, maskmap, vecmap)
        for i, data_dict in enumerate(self.train_loader):
            inputs = data_dict['img']
            maskmap = data_dict['maskmap']
            input_size = [inputs.size(3), inputs.size(2)]
            heatmap = self.heatmap_generator(data_dict['kpts'], input_size, maskmap=maskmap)
            vecmap = self.paf_generator(data_dict['kpts'], input_size, maskmap=maskmap)

            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs, heatmap, maskmap, vecmap = self.module_utilizer.to_device(inputs, heatmap, maskmap, vecmap)

            # Forward pass.
            paf_out, heatmap_out = self.pose_net(inputs)

            # Compute the loss of the train batch & backward.
            loss_heatmap = self.mse_loss(heatmap_out, heatmap, mask=maskmap, weights=self.weights)
            loss_associate = self.mse_loss(paf_out, vecmap, mask=maskmap, weights=self.weights)
            loss = loss_heatmap + loss_associate

            self.train_losses.update(loss.item(), inputs.size(0))
            self.train_loss_heatmap.update(loss_heatmap.item(), inputs.size(0))
            self.train_loss_associate.update(loss_associate.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Loss Heatmap:{}, Loss Asso: {}'.format(self.train_loss_heatmap.avg,
                                                                 self.train_loss_associate.avg))
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
                self.train_loss_heatmap.reset()
                self.train_loss_associate.reset()

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
            for i, data_dict in enumerate(self.val_loader):
                inputs = data_dict['img']
                maskmap = data_dict['maskmap']
                input_size = [inputs.size(3), inputs.size(2)]
                heatmap = self.heatmap_generator(data_dict['kpts'], input_size, maskmap=maskmap)
                vecmap = self.paf_generator(data_dict['kpts'], input_size, maskmap=maskmap)
                # Change the data type.
                inputs, heatmap, maskmap, vecmap = self.module_utilizer.to_device(inputs, heatmap, maskmap, vecmap)

                # Forward pass.
                paf_out, heatmap_out = self.pose_net(inputs)
                # Compute the loss of the val batch.
                loss_heatmap = self.mse_loss(heatmap_out[-1], heatmap, maskmap)
                loss_associate = self.mse_loss(paf_out[-1], vecmap, maskmap)
                loss = loss_heatmap + loss_associate

                self.val_losses.update(loss.item(), inputs.size(0))
                self.val_loss_heatmap.update(loss_heatmap.item(), inputs.size(0))
                self.val_loss_associate.update(loss_associate.item(), inputs.size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.module_utilizer.save_net(self.pose_net, save_mode='iters')
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
