#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualize the log files.


import re
import numpy as np
import matplotlib.pyplot as plt


class LogVisualizer(object):


    def vis_loss(self, log_file):

        with open(log_file, 'r') as file_stream:
            train_ax = list()
            train_ay = list()
            test_ax = list()
            test_ay = list()
            test_mark = 0

            for line in file_stream.readlines():
                if 'Iteration' in line:
                    m = re.match(r'.*Iteration:(.*)Learning.*', line)
                    iter = int(m.group(1))
                    train_ax.append(iter)
                    test_mark = iter

                elif 'TrainLoss' in line:
                    m = re.match(r'.*TrainLoss = (.*)', line)
                    loss = float(m.group(1))
                    train_ay.append(loss)

                elif 'TestLoss' in line:
                    m = re.match(r'.*TestLoss = (.*)', line)
                    loss = float(m.group(1))
                    test_ax.append(test_mark)
                    test_ay.append(loss)

                else:
                    continue

        train_ax = np.array(train_ax)
        train_ay = np.array(train_ay)
        test_ax = np.array(test_ax)
        test_ay = np.array(test_ay)
        plt.plot(train_ax, train_ay, label='Train Loss')
        plt.plot(test_ax, test_ay, label='Test Loss')
        plt.legend()
        plt.show()

    def vis_acc(self, log_file):
        with open(log_file, 'r') as file_stream:
            acc_ax = list()
            acc_ay = list()
            test_mark = 0

            for line in file_stream.readlines():
                if 'Iteration' in line and 'Train' in line:
                    m = re.match(r'.*Iteration:(.*)Learning.*', line)
                    iter = int(m.group(1))
                    test_mark = iter

                if 'Accuracy' in line:
                    m = re.match(r'.*Accuracy = (.*)', line)
                    loss = float(m.group(1))
                    acc_ax.append(test_mark)
                    acc_ay.append(loss)

                else:
                    continue

        plt.plot(acc_ax, acc_ay, label='Acc')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    #if len(sys.argv) != 2:
    #    print >> sys.stderr, "Need one args: log_file"
    #    exit(0)

    log_visualizer = LogVisualizer()
    log_visualizer.vis_loss('../../log/cls/fc_flower_cls.log')
