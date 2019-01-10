#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some methods used by main methods.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.optim import SGD, Adam, lr_scheduler

from utils.tools.logger import Logger as Log


class Trainer(object):

    @staticmethod
    def init(runner, net_params):
        optimizer = None
        if runner.configer.get('optim', 'optim_method') == 'sgd':
            optimizer = SGD(net_params,
                            lr=runner.configer.get('lr', 'base_lr'),
                            momentum=runner.configer.get('optim', 'sgd')['momentum'],
                            weight_decay=runner.configer.get('optim', 'sgd')['weight_decay'],
                            nesterov=runner.configer.get('optim', 'sgd')['nesterov'])

        elif runner.configer.get('optim', 'optim_method') == 'adam':
            optimizer = Adam(net_params,
                             lr=runner.configer.get('lr', 'base_lr'),
                             betas=runner.configer.get('optim', 'adam')['betas'],
                             eps=runner.configer.get('optim', 'adam')['eps'],
                             weight_decay=runner.configer.get('optim', 'adam')['weight_decay'])

        else:
            Log.error('Optimizer {} is not valid.'.format(runner.configer.get('optim', 'optim_method')))
            exit(1)

        policy = runner.configer.get('lr', 'lr_policy')

        scheduler = None
        if policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                            runner.configer.get('lr', 'step')['step_size'],
                                            gamma=runner.configer.get('lr', 'step')['gamma'])

        elif policy == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 runner.configer.get('lr', 'multistep')['stepvalue'],
                                                 gamma=runner.configer.get('lr', 'multistep')['gamma'])

        elif policy == 'lambda_poly':
            if runner.configer.get('lr', 'metric') == 'epoch':
                lambda_poly = lambda epoch: pow((1.0 - epoch / runner.configer.get('solver', 'max_epoch')), 0.9)
            else:
                assert runner.configer.get('lr', 'metric') == 'iters'
                lambda_poly = lambda epoch: pow((1.0 - epoch / runner.configer.get('solver', 'max_iters')), 0.9)

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)

        elif policy == 'lambda_linear':
            if runner.configer.get('lr', 'metric') == 'epoch':
                lambda_linear = lambda epoch: 1.0 - (epoch / runner.configer.get('solver', 'max_epoch'))
            else:
                assert runner.configer.get('lr', 'metric') == 'iters'
                lambda_linear = lambda epoch: 1.0 - (epoch / runner.configer.get('solver', 'max_iters'))

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_linear)

        elif policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode=runner.configer.get('lr', 'plateau')['mode'],
                                                       factor=runner.configer.get('lr', 'plateau')['factor'],
                                                       patience=runner.configer.get('lr', 'plateau')['patience'],
                                                       threshold=runner.configer.get('lr', 'plateau')['threshold'],
                                                       threshold_mode=runner.configer.get('lr', 'plateau')['thre_mode'],
                                                       cooldown=runner.configer.get('lr', 'plateau')['cooldown'],
                                                       min_lr=runner.configer.get('lr', 'plateau')['min_lr'],
                                                       eps=runner.configer.get('lr', 'plateau')['eps'])

        else:
            Log.error('Policy:{} is not valid.'.format(policy))
            exit(1)

        return optimizer, scheduler

    @staticmethod
    def update(runner, backbone_list=()):
        if not runner.configer.exists('lr', 'is_warm') or not runner.configer.get('lr', 'is_warm'):
            if runner.configer.get('lr', 'metric') == 'epoch':
                runner.scheduler.step(runner.runner_state['epoch'])
            else:
                assert runner.configer.get('lr', 'metric') == 'iters'
                runner.scheduler.step(runner.runner_state['iters'])

            return

        if runner.runner_state['iters'] < runner.configer.get('lr', 'warm')['warm_iters']:
            if runner.configer.get('lr', 'warm')['freeze_backbone']:
                for backbone_index in backbone_list:
                    runner.optimizer.param_groups[backbone_index]['lr'] = 0.0

            else:
                lr_ratio = (runner.runner_state['iters'] + 1) / runner.configer.get('lr', 'warm')['warm_iters']

                base_lr_list = runner.scheduler.get_lr()
                for param_group, base_lr in zip(runner.optimizer.param_groups, base_lr_list):
                    param_group['lr'] = base_lr * (lr_ratio ** 4)

        elif runner.runner_state['iters'] == runner.configer.get('lr', 'warm')['warm_iters']:
            try:
                base_lr_list = runner.scheduler.get_lr()
                for param_group, base_lr in zip(runner.optimizer.param_groups, base_lr_list):
                    param_group['lr'] = base_lr

            except AttributeError:
                nbb_lr = runner.configer.get('lr', 'base_lr') * runner.configer.get('lr', 'nbb_mult')
                for i, param_group in enumerate(runner.optimizer.param_groups):
                    if i in backbone_list:
                        continue

                    param_group[i]['lr'] = nbb_lr

        else:
            if runner.configer.get('lr', 'metric') == 'epoch':
                runner.scheduler.step(runner.runner_state['epoch'])
            else:
                assert runner.configer.get('lr', 'metric') == 'iters'
                runner.scheduler.step(runner.runner_state['iters'])







