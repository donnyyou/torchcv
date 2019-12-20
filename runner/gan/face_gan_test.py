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

from lib.runner.blob_helper import BlobHelper
from lib.runner.runner_helper import RunnerHelper
from model.gan.model_manager import ModelManager
from data.test.test_data_loader import TestDataLoader
from lib.tools.helper.dc_helper import DCHelper
from lib.tools.helper.image_helper import ImageHelper
from lib.tools.util.logger import Logger as Log


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
        if self.configer.exists('test', 'mode') and self.configer.get('test', 'mode') == 'nir2vis':
            jsonA_path = os.path.join(test_dir, 'val_label{}A.json'.format(self.configer.get('data', 'tag')))
            test_loader_A = self.test_loader.get_testloader(json_path=jsonA_path) if os.path.exists(jsonA_path) else None
            jsonB_path = os.path.join(test_dir, 'val_label{}B.json'.format(self.configer.get('data', 'tag')))
            test_loader_B = self.test_loader.get_testloader(json_path=jsonB_path) if os.path.exists(jsonB_path) else None

        else:
            test_loader_A, test_loader_B = None, None
            Log.error('Test Mode not Exists!')
            exit(1)

        assert test_loader_A is not None and test_loader_B is not None
        probe_features = []
        gallery_features = []
        probe_labels = []
        gallery_labels = []
        for data_dict in test_loader_A:
            new_data_dict = dict(imgA=data_dict['img'])
            with torch.no_grad():
                out_dict = self.gan_net(new_data_dict, testing=True)

            meta_list = DCHelper.tolist(data_dict['meta'])

            for idx in range(len(meta_list)):
                probe_features.append(out_dict['featA'][idx].cpu().numpy())
                probe_labels.append(meta_list[idx]['label'])

            for key, value in out_dict.items():
                for i in range(len(value)):
                    if 'feat' in key:
                        continue

                    img_bgr = self.blob_helper.tensor2bgr(value[i])
                    img_path = meta_list[i]['img_path']
                    Log.info('Image Path: {}'.format(img_path))
                    img_bgr = ImageHelper.resize(img_bgr,
                                                 target_size=self.configer.get('test', 'out_size'),
                                                 interpolation='linear')
                    ImageHelper.save(img_bgr, os.path.join(out_dir, key, meta_list[i]['filename']))

        for data_dict in test_loader_B:
            new_data_dict = dict(imgB=data_dict['img'])
            with torch.no_grad():
                out_dict = self.gan_net(new_data_dict, testing=True)

            meta_list = DCHelper.tolist(data_dict['meta'])

            for idx in range(len(meta_list)):
                gallery_features.append(out_dict['feat'][idx].cpu().numpy())
                gallery_labels.append(meta_list[idx]['label'])

            for key, value in out_dict.items():
                for i in range(len(value)):
                    if 'feat' in key:
                        continue

                    img_bgr = self.blob_helper.tensor2bgr(value[i])
                    img_path = meta_list[i]['img_path']
                    Log.info('Image Path: {}'.format(img_path))
                    img_bgr = ImageHelper.resize(img_bgr,
                                                 target_size=self.configer.get('test', 'out_size'),
                                                 interpolation='linear')
                    ImageHelper.save(img_bgr, os.path.join(out_dir, key, meta_list[i]['filename']))

        r_acc, tpr = self.decode(probe_features, gallery_features, probe_labels, gallery_labels)
        Log.info('Final Rank1 accuracy is {}'.format(r_acc))
        Log.info('Final VR@FAR=0.1% accuracy is {}'.format(tpr))

    @staticmethod
    def decode(probe_features, gallery_features, probe_labels, gallery_labels):
        probe_features = np.array(probe_features)
        gallery_features = np.array(gallery_features)
        score = cosine_similarity(gallery_features, probe_features).T
        # print('score.shape =', score.shape)
        # print('probe_names =', np.array(probe_names).shape)
        # print('gallery_names =', np.array(gallery_names).shape)
        print('===> compute metric')
        # print(probe_names[1], type(probe_names[1]))
        # exit()
        label = np.zeros_like(score)
        maxIndex = np.argmax(score, axis=1)
        # print('len = ', len(maxIndex))
        count = 0
        for i in range(len(maxIndex)):
            probe_names_repeat = np.repeat([probe_labels[i]], len(gallery_labels), axis=0).T
            # compare two string list
            result = np.equal(probe_names_repeat, gallery_labels) * 1
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
            if np.equal(int(probe_labels[i]), int(gallery_labels[maxIndex[i]])):
                count += 1
            else:
                pass
                # print(probe_img_list[i], gallery_img_list[ind])

        r_acc = count/(len(probe_labels)+1e-5)
        fpr, tpr, thresholds = roc_curve(label.flatten(), score.flatten())
        # print("In sub_experiment", label.size(0), 'count of true label :', count)
        # print('rank1 accuracy =', r_acc)
        # print('VR@FAR=0.1% accuracy =', tpr[fpr <= 0.001][-1])

        # plot_roc(fpr, tpr, thresholds, g_count)
        return r_acc, tpr[fpr <= 0.001][-1]
