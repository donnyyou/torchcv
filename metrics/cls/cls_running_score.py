#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Image classification running score.


from utils.tools.average_meter import AverageMeter


class ClsRunningScore(object):
    def __init__(self, configer):
        self.configer = configer
        self.top1_acc = AverageMeter()
        self.top3_acc = AverageMeter()
        self.top5_acc = AverageMeter()

    def get_top1_acc(self):
        return self.top1_acc.avg

    def get_top3_acc(self):
        return self.top3_acc.avg

    def get_top5_acc(self):
        return self.top5_acc.avg

    def update(self, output, target):
        """Computes the precision@k for the specified values of k"""
        topk = (1, 3, 5)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=False)
            res.append(correct_k / batch_size)

        self.top1_acc.update(res[0].item(), batch_size)
        self.top3_acc.update(res[1].item(), batch_size)
        self.top5_acc.update(res[2].item(), batch_size)

    def reset(self):
        self.top1_acc.reset()
        self.top3_acc.reset()
        self.top5_acc.reset()
