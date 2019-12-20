#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


import cv2
import time
import numpy as np
import torch

from data.seg.data_loader import DataLoader
from lib.runner.runner_helper import RunnerHelper
from lib.runner.trainer import Trainer
from model.seg.model_manager import ModelManager
from lib.tools.util.average_meter import AverageMeter, DictAverageMeter
from lib.tools.util.logger import Logger as Log
from lib.tools.helper.dc_helper import DCHelper
from metric.seg.seg_running_score import SegRunningScore
from lib.tools.vis.seg_visualizer import SegVisualizer


class FCNSegmentor(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = DictAverageMeter()
        self.val_losses = DictAverageMeter()
        self.seg_running_score = SegRunningScore(configer)
        self.seg_visualizer = SegVisualizer(configer)
        self.seg_model_manager = ModelManager(configer)
        self.seg_data_loader = DataLoader(configer)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.runner_state = dict()

        self._init_model()

    def _init_model(self):
        self.seg_net = self.seg_model_manager.get_seg_model()
        self.seg_net = RunnerHelper.load_net(self, self.seg_net)

        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.configer.get('solver'))

        self.train_loader = self.seg_data_loader.get_trainloader()
        self.val_loader = self.seg_data_loader.get_valloader()

        self.loss = self.seg_model_manager.get_seg_loss()

    def _get_parameters(self):
        lr_1 = []
        lr_10 = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                lr_10.append(value)
            else:
                lr_1.append(value)

        params = [{'params': lr_1, 'lr': self.configer.get('solver', 'lr')['base_lr']},
                  {'params': lr_10, 'lr': self.configer.get('solver', 'lr')['base_lr'] * 1.0}]
        return params

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.

        for i, data_dict in enumerate(self.train_loader):
            Trainer.update(self, warm_list=(0,), solver_dict=self.configer.get('solver'))
            self.data_time.update(time.time() - start_time)

            # Forward pass.
            data_dict = RunnerHelper.to_device(self, data_dict)
            out = self.seg_net(data_dict)
            # Compute the loss of the train batch & backward.
            loss_dict = self.loss(out)
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

            if self.runner_state['iters'] % self.configer.get('solver.save_iters') == 0 \
                    and self.configer.get('local_rank') == 0:
                RunnerHelper.save_net(self, self.seg_net)

            if self.configer.get('solver', 'lr')['metric'] == 'iters' \
                    and self.runner_state['iters'] == self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            if self.runner_state['iters'] % self.configer.get('solver', 'test_interval') == 0 \
                    and not self.configer.get('network.distributed'):
                self.val()

        self.runner_state['epoch'] += 1

    def val(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        start_time = time.time()

        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            data_dict = RunnerHelper.to_device(self, data_dict)
            with torch.no_grad():
                # Forward pass.
                out = self.seg_net(data_dict)
                loss_dict = self.loss(out)
                # Compute the loss of the val batch.
                out_dict, _ = RunnerHelper.gather(self, out)
            
            self.val_losses.update({key: loss.item() for key, loss in loss_dict.items()}, data_dict['img'].size(0))
            self._update_running_score(out_dict['out'], DCHelper.tolist(data_dict['meta']))

            # Update the vars of the val phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.runner_state['performance'] = self.seg_running_score.get_mean_iou()
        self.runner_state['val_loss'] = self.val_losses.avg['loss']
        RunnerHelper.save_net(self, self.seg_net,
                              performance=self.seg_running_score.get_mean_iou(),
                              val_loss=self.val_losses.avg['loss'])

        # Print the log info & reset the states.
        Log.info(
            'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
            'Loss = {0}\n'.format(self.val_losses.info(), batch_time=self.batch_time))
        Log.info('Mean IOU: {}\n'.format(self.seg_running_score.get_mean_iou()))
        Log.info('Pixel ACC: {}\n'.format(self.seg_running_score.get_pixel_acc()))
        self.batch_time.reset()
        self.val_losses.reset()
        self.seg_running_score.reset()
        self.seg_net.train()

    def _update_running_score(self, pred, metas):
        pred = pred.permute(0, 2, 3, 1)
        for i in range(pred.size(0)):
            border_size = metas[i]['border_wh']
            ori_target = metas[i]['ori_target']
            total_logits = cv2.resize(pred[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(metas[i]['ori_img_wh']), interpolation=cv2.INTER_CUBIC)
            labelmap = np.argmax(total_logits, axis=-1)
            self.seg_running_score.update(labelmap[None], ori_target[None])


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
