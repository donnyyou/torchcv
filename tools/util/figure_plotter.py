#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
"""Plot Figures from papers"""


import os
from matplotlib import pyplot
import matplotlib.pyplot as plt


class FigurePlotter(object):

    @staticmethod
    def plot_line(x_list, y_list, name_list, style_list, save_fig=False):
        for x, y, name, style in zip(x_list, y_list, name_list, style_list):
            plt.plot(x, y, style, label=name)

        plt.legend()  # 让图例生效
        plt.grid(linestyle="--")
        plt.xticks(x_list[0], rotation=1)
        plt.ylim(76.0, 79.0)
        # plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel('K Mixtures', fontsize=12)  # X轴标签
        plt.ylabel("Top1 Acc", fontsize=12)  # Y轴标签
        plt.show()
        # plt.title("A simple plot") #标题
        # plt.savefig('./f1.jpg', dpi=900)


if __name__ == "__main__":
    x_list = [[5, 10, 15, 20, 25], [5, 10, 15, 20, 25]]
    y_list = [[77.84, 77.84, 77.84, 77.84, 77.84], [77.83, 77.90, 78.24, 77.76, 77.74]]
    name_list = ['SENet-50', 'kSENet-50']
    FigurePlotter.plot_line(x_list, y_list, name_list, style_list=['o:', 'v-'])
