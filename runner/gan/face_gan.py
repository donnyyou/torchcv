#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for GAN.


import time
import torch

from data.gan.data_loader import DataLoader
from lib.runner.runner_helper import RunnerHelper
from lib.runner.trainer import Trainer
from runner.gan.face_gan_test import FaceGANTest
from model.gan.model_manager import ModelManager
from lib.tools.util.average_meter import AverageMeter
from lib.tools.helper.dc_helper import DCHelper
from lib.tools.util.logger import Logger as Log


class FaceGAN(object):
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

        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.configer.get('solver'))

        self.train_loader = self.seg_data_loader.get_trainloader()
        self.val_loader = self.seg_data_loader.get_valloader()

    def _get_parameters(self):

        return self.gan_net.parameters()

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.gan_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        for i, data_dict in enumerate(self.train_loader):
            Trainer.update(self, solver_dict=self.configer.get('solver'))
            self.data_time.update(time.time() - start_time)

            # Forward pass.
            out_dict = self.gan_net(data_dict)
            # outputs = self.module_utilizer.gather(outputs)
            loss = out_dict['loss'].mean()
            self.train_losses.update(loss.item(), len(DCHelper.tolist(data_dict['meta'])))
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

        self.runner_state['epoch'] += 1

    def val(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.gan_net.eval()
        start_time = time.time()

        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):

            with torch.no_grad():
                # Forward pass.
                out_dict = self.gan_net(data_dict)
                # Compute the loss of the val batch.

            self.val_losses.update(out_dict['loss'].mean().item(), len(DCHelper.tolist(data_dict['meta'])))
            meta_list = DCHelper.tolist(data_dict['meta'])
            probe_features = []
            gallery_features = []
            probe_labels = []
            gallery_labels = []
            for idx in range(len(meta_list)):
                gallery_features.append(out_dict['featB'][idx].cpu().numpy())
                gallery_labels.append(meta_list[idx]['labelB'])
                probe_features.append(out_dict['featA'][idx].cpu().numpy())
                probe_labels.append(meta_list[idx]['labelA'])

            rank_1, vr_far_001 = FaceGANTest.decode(probe_features, gallery_features, probe_labels, gallery_labels)
            Log.info('Rank1 accuracy is {}'.format(rank_1))
            Log.info('VR@FAR=0.1% accuracy is {}'.format(vr_far_001))
            # Update the vars of the val phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        RunnerHelper.save_net(self, self.gan_net,
                              val_loss=self.val_losses.avg)

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
