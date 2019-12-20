#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some runner used by main runner.


from torch.optim import SGD, Adam, lr_scheduler

from lib.tools.util.logger import Logger as Log


class Trainer(object):

    @staticmethod
    def init(net_params, solver_dict=None):
        optimizer = None
        optim_params = solver_dict['optim']
        if optim_params['optim_method'] == 'sgd':
            optimizer = SGD(net_params,
                            lr=solver_dict['lr']['base_lr'],
                            momentum=optim_params['sgd']['momentum'],
                            weight_decay=optim_params['sgd']['weight_decay'],
                            nesterov=optim_params['sgd']['nesterov'])

        elif optim_params['optim_method'] == 'adam':
            optimizer = Adam(net_params,
                             lr=solver_dict['lr']['base_lr'],
                             betas=optim_params['adam']['betas'],
                             eps=optim_params['adam']['eps'],
                             weight_decay=optim_params['adam']['weight_decay'])

        else:
            Log.error('Optimizer {} is not valid.'.format(optim_params['optim_method']))
            exit(1)

        lr_params = solver_dict['lr']
        scheduler = None
        if lr_params['lr_policy'] == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                            lr_params['step']['step_size'],
                                            gamma=lr_params['step']['gamma'])

        elif lr_params['lr_policy'] == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 lr_params['multistep']['stepvalue'],
                                                 gamma=lr_params['multistep']['gamma'])

        elif lr_params['lr_policy'] == 'lambda_poly':
            if lr_params['metric'] == 'epoch':
                lambda_poly = lambda epoch: pow((1.0 - epoch / solver_dict['max_epoch']),
                                                lr_params['lambda_poly']['power'])
            else:
                assert lr_params['metric'] == 'iters'
                lambda_poly = lambda epoch: pow((1.0 - epoch / solver_dict['max_iters']),
                                                lr_params['lambda_poly']['power'])

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)

        elif lr_params['lr_policy'] == 'lambda_range':
            if lr_params['metric'] == 'epoch':
                lambda_poly = lambda epoch: pow((1.0 - epoch / solver_dict['max_epoch']),
                                                lr_params['lambda_range']['max_power'] * epoch / solver_dict['max_epoch'])
            else:
                assert lr_params['metric'] == 'iters'
                lambda_poly = lambda epoch: pow((1.0 - epoch / solver_dict['max_iters']),
                                                lr_params['lambda_range']['max_power'] * epoch / solver_dict['max_iters'])

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)

        elif lr_params['lr_policy'] == 'lambda_linear':
            if lr_params['metric'] == 'epoch':
                lambda_linear = lambda epoch: 1.0 - (epoch / solver_dict['max_epoch'])
            else:
                assert lr_params['metric'] == 'iters'
                lambda_linear = lambda epoch: 1.0 - (epoch / solver_dict['max_iters'])

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_linear)

        elif lr_params['lr_policy'] == 'lambda_fixlinear':
            lambda_fixlinear = lambda epoch: max(0.0, 1.0 - (max(0, epoch - lr_params['lambda_fixlinear']['fix_value'])
                                                             / lr_params['lambda_fixlinear']['linear_value']))
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fixlinear)

        elif lr_params['lr_policy'] == 'lambda_cosine':
            import math
            lambda_cosine = lambda iters: (1 + math.cos(math.pi * iters / solver_dict['max_iters'])) / 2
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)

        elif lr_params['lr_policy'] == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode=lr_params['plateau']['mode'],
                                                       factor=lr_params['plateau']['factor'],
                                                       patience=lr_params['plateau']['patience'],
                                                       threshold=lr_params['plateau']['threshold'],
                                                       threshold_mode=lr_params['plateau']['thre_mode'],
                                                       cooldown=lr_params['plateau']['cooldown'],
                                                       min_lr=lr_params['plateau']['min_lr'],
                                                       eps=lr_params['plateau']['eps'])

        else:
            Log.error('Policy:{} is not valid.'.format(lr_params['lr_policy']))
            exit(1)

        return optimizer, scheduler

    @staticmethod
    def update(runner, warm_list=(), warm_lr_list=None, solver_dict=None):
        if solver_dict['lr']['metric'] == 'epoch':
            if runner.runner_state['last_epoch'] != runner.runner_state['epoch']:
                runner.scheduler.step(runner.runner_state['epoch'])
                runner.runner_state['last_epoch'] = runner.runner_state['epoch']
        else:
            assert solver_dict['lr']['metric'] == 'iters'
            if runner.runner_state['last_iters'] != runner.runner_state['iters']:
                runner.scheduler.step(runner.runner_state['iters'])
                runner.runner_state['last_iters'] = runner.runner_state['iters']

        if 'is_warm' in solver_dict['lr'] and solver_dict['lr']['is_warm']:
            if runner.runner_state['iters'] < solver_dict['lr']['warm']['warm_iters']:
                if solver_dict['lr']['warm']['freeze_backbone']:
                    for group_index in warm_list:
                        runner.optimizer.param_groups[group_index]['lr'] = 0.0

                else:
                    lr_ratio = (runner.runner_state['iters'] + 1) / solver_dict['lr']['warm']['warm_iters']
                    lr_ratio = lr_ratio ** solver_dict['lr']['warm']['power']
                    for group_index, base_lr in zip(warm_list, warm_lr_list):
                        runner.optimizer.param_groups[group_index]['lr'] = base_lr * lr_ratio

            elif runner.runner_state['iters'] == solver_dict['lr']['warm']['warm_iters']:
                for group_index, base_lr in zip(warm_list, warm_lr_list):
                    runner.optimizer.param_groups[group_index]['lr'] = base_lr
