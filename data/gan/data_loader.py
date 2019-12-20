#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from torch.utils import data

from data.gan.datasets.default_pix2pix_dataset import DefaultPix2pixDataset
from data.gan.datasets.default_cyclegan_dataset import DefaultCycleGANDataset
from data.gan.datasets.default_facegan_dataset import DefaultFaceGANDataset
import lib.data.pil_aug_transforms as pil_aug_trans
import lib.data.cv2_aug_transforms as cv2_aug_trans
import lib.data.transforms as trans
from lib.data.collate import collate
from lib.tools.util.logger import Logger as Log


class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_train_transform = pil_aug_trans.PILAugCompose(self.configer, split='train')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='train')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_val_transform = pil_aug_trans.PILAugCompose(self.configer, split='val')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_val_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='val')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(**self.configer.get('data', 'normalize')), ])

        self.label_transform = trans.Compose([
            trans.ToLabel(),
            trans.ReLabel(255, -1), ])

    def get_trainloader(self):
        if self.configer.get('dataset', default=None) == 'default_pix2pix':
            dataset = DefaultPix2pixDataset(root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                                            aug_transform=self.aug_train_transform,
                                            img_transform=self.img_transform,
                                            configer=self.configer)

        elif self.configer.get('dataset') == 'default_cyclegan':
            dataset = DefaultCycleGANDataset(root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                                             aug_transform=self.aug_train_transform,
                                             img_transform=self.img_transform,
                                             configer=self.configer)

        elif self.configer.get('dataset') == 'default_facegan':
            dataset = DefaultFaceGANDataset(root_dir=self.configer.get('data', 'data_dir'),
                                            dataset='train', tag=self.configer.get('data', 'tag'),
                                            aug_transform=self.aug_train_transform,
                                            img_transform=self.img_transform,
                                            configer=self.configer)

        else:
            Log.error('{} train loader is invalid.'.format(self.configer.get('train', 'loader')))
            exit(1)

        trainloader = data.DataLoader(
            dataset,
            batch_size=self.configer.get('train', 'batch_size'), shuffle=True,
            num_workers=self.configer.get('data', 'workers'), pin_memory=True,
            drop_last=self.configer.get('data', 'drop_last'),
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('train', 'data_transformer')
            )
        )

        return trainloader



    def get_valloader(self, dataset=None):
        dataset = 'val' if dataset is None else dataset
        if self.configer.get('dataset') == 'default_pix2pix':
            dataset = DefaultPix2pixDataset(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                                            aug_transform=self.aug_val_transform,
                                            img_transform=self.img_transform,
                                            configer=self.configer)

        elif self.configer.get('dataset') == 'default_cyclegan':
            dataset = DefaultCycleGANDataset(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                                             aug_transform=self.aug_val_transform,
                                             img_transform=self.img_transform,
                                             configer=self.configer)

        elif self.configer.get('dataset') == 'default_facegan':
            dataset = DefaultFaceGANDataset(root_dir=self.configer.get('data', 'data_dir'),
                                            dataset=dataset, tag=self.configer.get('data', 'tag'),
                                            aug_transform=self.aug_val_transform,
                                            img_transform=self.img_transform,
                                            configer=self.configer)
    
        else:
            Log.error('{} val loader is invalid.'.format(self.configer.get('val', 'loader')))
            exit(1)

        valloader = data.DataLoader(
            dataset,
            batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
            num_workers=self.configer.get('data', 'workers'), pin_memory=True,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('val', 'data_transformer')
            )
        )

        return valloader


if __name__ == "__main__":
    # Test data loader.
    pass
