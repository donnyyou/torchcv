#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for GAN.


import time
import torch

from data.gan.data_loader import DataLoader
from lib.runner.runner_helper import RunnerHelper
from lib.runner.trainer import Trainer
from model.gan.model_manager import ModelManager
from lib.tools.util.average_meter import AverageMeter
from lib.tools.helper.dc_helper import DCHelper
from lib.tools.util.logger import Logger as Log


class ImageTranslator(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.model_manager = ModelManager(configer)
        self.seg_data_loader = DataLoader(configer)

        self.gan_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        self.gan_net = self.model_manager.gan_model()
        self.gan_net = RunnerHelper.load_net(self, self.gan_net)

        self.optimizer_G, self.scheduler_G = Trainer.init(self._get_parameters()[0], self.configer.get('solver'))
        self.optimizer_D, self.scheduler_D = Trainer.init(self._get_parameters()[1], self.configer.get('solver'))

        self.train_loader = self.seg_data_loader.get_trainloader()
        self.val_loader = self.seg_data_loader.get_valloader()

    def _get_parameters(self):
        params_G = []
        params_D = []
        params_dict = dict(self.gan_net.named_parameters())
        for key, value in params_dict.items():
            if 'G' not in key:
                params_D.append(value)
            else:
                params_G.append(value)

        return params_G, params_D

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.gan_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.scheduler_G.step(self.runner_state['epoch'])
        self.scheduler_D.step(self.runner_state['epoch'])
        for i, data_dict in enumerate(self.train_loader):
            self.data_time.update(time.time() - start_time)

            # Forward pass.
            out_dict = self.gan_net(data_dict)
            # outputs = self.module_utilizer.gather(outputs)
            self.optimizer_G.zero_grad()
            loss_G = out_dict['loss_G'].mean()
            loss_G.backward()
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            loss_D = out_dict['loss_D'].mean()
            loss_D.backward()
            self.optimizer_D.step()
            loss = loss_G + loss_D
            self.train_losses.update(loss.item(), len(DCHelper.tolist(data_dict['meta'])))

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
                         [RunnerHelper.get_lr(self.optimizer_G), RunnerHelper.get_lr(self.optimizer_D)],
                         batch_time=self.batch_time, data_time=self.data_time, loss=self.train_losses))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            if self.configer.get('solver', 'lr')['metric'] == 'iters' \
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
        self.gan_net.eval()
        start_time = time.time()

        for j, data_dict in enumerate(self.val_loader):
            with torch.no_grad():
                # Forward pass.
                out_dict = self.gan_net(data_dict)
                # Compute the loss of the val batch.

            self.val_losses.update(out_dict['loss_G'].mean().item() + out_dict['loss_D'].mean().item(),
                                   len(DCHelper.tolist(data_dict['meta'])))
            # Update the vars of the val phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        RunnerHelper.save_net(self, self.gan_net, val_loss=self.val_losses.avg)

        # Print the log info & reset the states.
        Log.info(
            'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
            'Loss {loss.avg:.8f}\n'.format(
                batch_time=self.batch_time, loss=self.val_losses))
        self.batch_time.reset()
        self.val_losses.reset()
        self.gan_net.train()


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
