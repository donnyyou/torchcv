#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from PIL import Image
import os
import json
import os.path
import numpy as np
import sys
import hashlib
import argparse
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


IMAGE_DIR = 'image'


class Cifar10ClsGenerator(object):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, args, image_dir=IMAGE_DIR):
        self.args = args
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.train_data = []
        self.train_targets = []
        self.test_data = []
        self.test_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in self.train_list:
            file_path = os.path.join(self.args.root_dir, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_targets.extend(entry['labels'])
                else:
                    self.train_targets.extend(entry['fine_labels'])

        self.train_data = np.vstack(self.train_data).reshape(-1, 3, 32, 32)
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

        for file_name, checksum in self.test_list:
            file_path = os.path.join(self.args.root_dir, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.test_data.append(entry['data'])
                if 'labels' in entry:
                    self.test_targets.extend(entry['labels'])
                else:
                    self.test_targets.extend(entry['fine_labels'])

        self.test_data = np.vstack(self.test_data).reshape(-1, 3, 32, 32)
        self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        self.train_json_file = os.path.join(self.args.save_dir, 'train/label.json')
        self.val_json_file = os.path.join(self.args.save_dir, 'val/label.json')

        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        self.train_image_dir = os.path.join(self.args.save_dir, 'train', image_dir)
        self.val_image_dir = os.path.join(self.args.save_dir, 'val', image_dir)
        if not os.path.exists(self.train_image_dir):
            os.makedirs(self.train_image_dir)

        if not os.path.exists(self.val_image_dir):
            os.makedirs(self.val_image_dir)

    def _load_meta(self):
        path = os.path.join(self.args.root_dir, self.meta['filename'])
        if not self.check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    @staticmethod
    def check_integrity(fpath, md5=None):
        if md5 is None:
            return True
        if not os.path.isfile(fpath):
            return False
        md5o = hashlib.md5()
        with open(fpath, 'rb') as f:
            # read in 1MB chunks
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                md5o.update(chunk)
        md5c = md5o.hexdigest()
        if md5c != md5:
            return False
        return True

    def generate_label(self):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        train_json_list = list()
        val_json_list = list()

        for index in range(len(self.train_targets)):
            img, target = self.train_data[index], self.train_targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            filename = str(index).zfill(len(str(len(self.train_targets))))
            filename = '{}.jpg'.format(filename)
            img.save(os.path.join(self.train_image_dir, filename))
            img_dict = dict()
            img_dict['image_path'] = '{}/{}'.format(IMAGE_DIR, filename)
            img_dict['label'] = target
            train_json_list.append(img_dict)

        for index in range(len(self.test_targets)):
            img, target = self.test_data[index], self.test_targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            filename = str(index).zfill(len(str(len(self.test_targets))))
            filename = '{}.jpg'.format(filename)
            img.save(os.path.join(self.val_image_dir, filename))
            img_dict = dict()
            img_dict['image_path'] = '{}/{}'.format(IMAGE_DIR, filename)
            img_dict['label'] = target
            val_json_list.append(img_dict)

        fw = open(self.train_json_file, 'w')
        fw.write(json.dumps(train_json_list))
        fw.close()

        fw = open(self.val_json_file, 'w')
        fw.write(json.dumps(val_json_list))
        fw.close()

    def _check_integrity(self):
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(self.args.root_dir, filename)
            if not self.check_integrity(fpath, md5):
                return False
        return True


class Cifar100ClsGenerator(Cifar10ClsGenerator):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--root_dir', default=None, type=str,
                        dest='root_dir', help='The directory of the image data.')
    parser.add_argument('--dataset', default="cifar10", type=str,
                        dest='dataset', help='The dataset name.')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        cifar10_cls_generator = Cifar10ClsGenerator(args)
        cifar10_cls_generator.generate_label()

    else:
        assert args.dataset == 'cifar100'
        cifar100_cls_generator = Cifar100ClsGenerator(args)
        cifar100_cls_generator.generate_label()
