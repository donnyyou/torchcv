#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for GAN.


import os
import glob
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve

from methods.tools.blob_helper import BlobHelper
from methods.tools.runner_helper import RunnerHelper
from models.gan.model_manager import ModelManager
from datasets.test.test_data_loader import TestDataLoader
from utils.helpers.dc_helper import DCHelper
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class FaceGANTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.blob_helper = BlobHelper(configer)
        self.model_manager = ModelManager(configer)
        self.test_loader = TestDataLoader(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.gan_net = None

        self._init_model()

    def _init_model(self):
        self.gan_net = self.model_manager.gan_model()
        self.gan_net = RunnerHelper.load_net(self, self.gan_net)
        self.gan_net.eval()

    def test(self, test_dir, out_dir):
        gallery_file_list = '*_gallery_*.txt'
        probe_file_list = '*_probe_*.txt'
        gallery_file_list = glob.glob(test_dir + '/' + gallery_file_list)
        probe_file_list = glob.glob(test_dir + '/' + probe_file_list)
        # remove *_dev.txt file in both list
        gallery_file_list = sorted(gallery_file_list)
        probe_file_list = sorted(probe_file_list)
        rank1_acc = []
        vr_acc = []
        for i in range(len(gallery_file_list)):
            probe_features = []
            gallery_features = []
            probe_names = []
            gallery_names = []
            Log.info('Gallery File: {}'.format(gallery_file_list[i]))
            for data_dict in self.test_loader.get_testloader(list_path=gallery_file_list[i]):
                new_data_dict = dict(gallery=data_dict['img'])
                out_dict = self.gan_net(new_data_dict)
                meta_list = DCHelper.tolist(data_dict['meta'])

                for idx in range(len(out_dict['feat'])):
                    gallery_features.append(out_dict['feat'][idx].cpu().numpy())
                    gallery_names.append(meta_list[idx]['img_path'].split("/")[-2])

            Log.info('Probe File: {}'.format(probe_file_list[i]))
            for data_dict in self.test_loader.get_testloader(list_path=probe_file_list[i]):
                new_data_dict = dict(probe=data_dict['img'])
                out_dict = self.gan_net(new_data_dict)
                meta_list = DCHelper.tolist(data_dict['meta'])

                for key, value in out_dict.item():
                    if 'feat' in key:
                        for idx in range(len(value)):
                            probe_features.append(value[idx].cpu().numpy())
                            probe_names.append(meta_list[idx]['img_path'].split("/")[-2])

                        continue
                    else:
                        for idx in range(len(value)):
                            img_bgr = self.blob_helper.tensor2bgr(value[idx])
                            filename = meta_list[idx]['img_path'].rstrip().split('/')[-1]
                            ImageHelper.save(img_bgr, os.path.join(out_dir, key, filename))

            probe_features = np.array(probe_features)
            gallery_features = np.array(gallery_features)
            score = cosine_similarity(gallery_features, probe_features).T
            r_acc, tpr = self.compute_metric(score, probe_names, gallery_names)
            # print('score={}, probe_names={}, gallery_names={}'.format(score, probe_names, gallery_names))
            rank1_acc.append(r_acc)
            vr_acc.append(tpr)

        avg_r_a = np.mean(np.array(rank1_acc))
        std_r_a = np.std(np.array(rank1_acc))
        avg_v_a = np.mean(np.array(vr_acc))
        std_v_a = np.std(np.array(vr_acc))
        # avg_vr_acc = sum(vr_acc)/(len(vr_acc) + 1e-5)
        print()
        print('=====================================================')
        print('Final Rank1 accuracy is', avg_r_a * 100, "% +", std_r_a)
        print('Final VR@FAR=0.1% accuracy is', avg_v_a * 100, "% +", std_v_a)
        print('=====================================================')
        print()
        return avg_r_a, std_r_a, avg_v_a, std_v_a

    def compute_metric(self, score, probe_names, gallery_names):
        # print('score.shape =', score.shape)
        # print('probe_names =', np.array(probe_names).shape)
        # print('gallery_names =', np.array(gallery_names).shape)
        print('===> compute metrics')
        # print(probe_names[1], type(probe_names[1]))
        # exit()
        label = np.zeros_like(score)
        maxIndex = np.argmax(score, axis=1)
        # print('len = ', len(maxIndex))
        count = 0
        for i in range(len(maxIndex)):
            probe_names_repeat = np.repeat([probe_names[i]], len(gallery_names), axis=0).T
            # compare two string list
            result = np.equal(probe_names_repeat, gallery_names) * 1
            # result = np.core.defchararray.equal(probe_names_repeat, gallery_names) * 1
            # find the index of image in the gallery that has the same name as probe image
            # print(result)
            # print('++++++++++++++++++++++++++++++++=')
            index = np.nonzero(result==1)

            # if i == 10:
            #     exit()
            assert len(index[0]) == 1
            label[i][index[0][0]] = 1

            # find the max similarty score in gallery has the same name as probe image
            if np.equal(int(probe_names[i]), int(gallery_names[maxIndex[i]])):
                count += 1
            else:
                pass
                # print(probe_img_list[i], gallery_img_list[ind])

        r_acc = count/(len(probe_names)+1e-5)
        fpr, tpr, thresholds = roc_curve(label.flatten(), score.flatten())
        print("In sub_experiment", label.size(0), 'count of true label :', count)
        print('rank1 accuracy =', r_acc)
        print('VR@FAR=0.1% accuracy =', tpr[fpr <= 0.001][-1])

        # plot_roc(fpr, tpr, thresholds, g_count)
        return r_acc, tpr[fpr <= 0.001][-1]
