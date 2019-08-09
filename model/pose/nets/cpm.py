#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Pose Net fot Parts Detection.


import torch.nn as nn
import torch.nn.functional as F
import torch


class CPM(nn.Module):
    def __init__(self, configer):
        super(CPM, self).__init__()
        self.out_c = configer.get('network', 'heatmap_out')
        self.pool_center_lower = nn.AvgPool2d(kernel_size=9, stride=8)
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_stage1 = nn.Conv2d(512, self.out_c, kernel_size=1)

        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2,  padding=1)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage2 = nn.Conv2d(32 + self.out_c, 128, kernel_size=11, padding=5)
        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage2 = nn.Conv2d(128, self.out_c, kernel_size=1, padding=0)

        self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage3 = nn.Conv2d(32 + self.out_c, 128, kernel_size=11, padding=5)
        self.Mconv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage3 = nn.Conv2d(128, self.out_c, kernel_size=1, padding=0)

        self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage4 = nn.Conv2d(32 + self.out_c, 128, kernel_size=11, padding=5)
        self.Mconv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage4 = nn.Conv2d(128, self.out_c, kernel_size=1, padding=0)

        self.conv1_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage5 = nn.Conv2d(32 + self.out_c, 128, kernel_size=11, padding=5)
        self.Mconv2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage5 = nn.Conv2d(128, self.out_c, kernel_size=1, padding=0)

        self.conv1_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage6 = nn.Conv2d(32 + self.out_c, 128, kernel_size=11, padding=5)
        self.Mconv2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage6 = nn.Conv2d(128, self.out_c, kernel_size=1, padding=0)

    def _stage1(self, image):
        """
        Output result of stage 1
        :param image: source image with (368, 368)
        :return: conv7_stage1_map
        """
        x = self.pool1_stage1(F.relu(self.conv1_stage1(image)))
        x = self.pool2_stage1(F.relu(self.conv2_stage1(x)))
        x = self.pool3_stage1(F.relu(self.conv3_stage1(x)))
        x = F.relu(self.conv4_stage1(x))
        x = F.relu(self.conv5_stage1(x))
        x = F.relu(self.conv6_stage1(x))
        x = self.conv7_stage1(x)

        return x

    def _middle(self, image):
        """
        Compute shared pool3_stage_map for the following stage
        :param image: source image with (368, 368)
        :return: pool3_stage2_map
        """
        x = self.pool1_stage2(F.relu(self.conv1_stage2(image)))
        x = self.pool2_stage2(F.relu(self.conv2_stage2(x)))
        x = self.pool3_stage2(F.relu(self.conv3_stage2(x)))

        return x

    def _stage2(self, pool3_stage2_map, conv7_stage1_map):
        """
        Output result of stage 2
        :param pool3_stage2_map
        :param conv7_stage1_map
        :return: Mconv5_stage2_map
        """
        x = F.relu(self.conv4_stage2(pool3_stage2_map))
        x = torch.cat([x, conv7_stage1_map], dim=1)
        x = F.relu(self.Mconv1_stage2(x))
        x = F.relu(self.Mconv2_stage2(x))
        x = F.relu(self.Mconv3_stage2(x))
        x = F.relu(self.Mconv4_stage2(x))
        x = self.Mconv5_stage2(x)

        return x

    def _stage3(self, pool3_stage2_map, Mconv5_stage2_map):
        """
        Output result of stage 3
        :param pool3_stage2_map:
        :param Mconv5_stage2_map:
        :return: Mconv5_stage3_map
        """
        x = F.relu(self.conv1_stage3(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage2_map], dim=1)
        x = F.relu(self.Mconv1_stage3(x))
        x = F.relu(self.Mconv2_stage3(x))
        x = F.relu(self.Mconv3_stage3(x))
        x = F.relu(self.Mconv4_stage3(x))
        x = self.Mconv5_stage3(x)

        return x

    def _stage4(self, pool3_stage2_map, Mconv5_stage3_map):
        """
        Output result of stage 4
        :param pool3_stage2_map:
        :param Mconv5_stage3_map:
        :return:Mconv5_stage4_map
        """
        x = F.relu(self.conv1_stage4(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage3_map], dim=1)
        x = F.relu(self.Mconv1_stage4(x))
        x = F.relu(self.Mconv2_stage4(x))
        x = F.relu(self.Mconv3_stage4(x))
        x = F.relu(self.Mconv4_stage4(x))
        x = self.Mconv5_stage4(x)

        return x

    def _stage5(self, pool3_stage2_map, Mconv5_stage4_map):
        """
        Output result of stage 5
        :param pool3_stage2_map:
        :param Mconv5_stage4_map:
        :return:Mconv5_stage5_map
        """
        x = F.relu(self.conv1_stage5(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage4_map], dim=1)
        x = F.relu(self.Mconv1_stage5(x))
        x = F.relu(self.Mconv2_stage5(x))
        x = F.relu(self.Mconv3_stage5(x))
        x = F.relu(self.Mconv4_stage5(x))
        x = self.Mconv5_stage5(x)

        return x

    def _stage6(self, pool3_stage2_map, Mconv5_stage5_map):
        """
        Output result of stage 6
        :param pool3_stage2_map:
        :param Mconv5_stage6_map:
        :param pool_center_lower_map:
        :return:Mconv5_stage6_map
        """
        x = F.relu(self.conv1_stage6(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage5_map], dim=1)
        x = F.relu(self.Mconv1_stage6(x))
        x = F.relu(self.Mconv2_stage6(x))
        x = F.relu(self.Mconv3_stage6(x))
        x = F.relu(self.Mconv4_stage6(x))
        x = self.Mconv5_stage6(x)

        return x

    def forward(self, image):
        conv7_stage1_map = self._stage1(image)  # result of stage 1

        pool3_stage2_map = self._middle(image)

        Mconv5_stage2_map = self._stage2(pool3_stage2_map, conv7_stage1_map)  # result of stage 2
        Mconv5_stage3_map = self._stage3(pool3_stage2_map, Mconv5_stage2_map)  # result of stage 3
        Mconv5_stage4_map = self._stage4(pool3_stage2_map, Mconv5_stage3_map)  # result of stage 4
        Mconv5_stage5_map = self._stage5(pool3_stage2_map, Mconv5_stage4_map)  # result of stage 5
        Mconv5_stage6_map = self._stage6(pool3_stage2_map, Mconv5_stage5_map)  # result of stage 6

        return [conv7_stage1_map, Mconv5_stage2_map, Mconv5_stage3_map,
                Mconv5_stage4_map, Mconv5_stage5_map, Mconv5_stage6_map]
