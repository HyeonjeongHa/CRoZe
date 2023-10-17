
# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import sys
import hashlib
from PIL import Image
import torch
import torchvision
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
if sys.version_info[0] == 2:
  import cPickle as pickle
else:
  import pickle
  
from .corrupt_dataset import *
from .sampler import *


def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


def get_dataloaders(
    train_batch_size, 
    test_batch_size, 
    dataset, 
    num_workers, 
    resize=None, 
    datadir='_dataset', 
    normalize=True,
    return_ds=False,
    corruption=None,
    severity=None
):

    if 'ImageNet16' in dataset:
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
        size, pad = 16, 2
    elif 'cifar' in dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size, pad = 32, 4
    elif 'svhn' in dataset:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        size, pad = 32, 0
    elif dataset == 'ImageNet1k':
        from .h5py_dataset import H5Dataset
        size, pad = 224, 2
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        #resize = 256

    if resize is None:
        resize = size

    train_transform_list = [
        transforms.RandomCrop(size, padding=pad),
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    test_transform_list = [
        transforms.Resize(resize),
        transforms.ToTensor(),
    ]
    
    if normalize:
        train_transform_list.append(transforms.Normalize(mean,std))
        test_transform_list.append(transforms.Normalize(mean,std))
    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)
        
    if dataset == 'cifar10':
        if corruption is not None:
            train_dataset = None
            test_dataset = CommonCorruptionDataset(datadir, dataset, corruption, test_transform, severity)
        else:
            train_dataset = CIFAR10(datadir, True, train_transform, download=True)
            test_dataset = CIFAR10(datadir, False, test_transform, download=True)
    elif dataset == 'cifar100':
        if corruption is not None:
            train_dataset = None
            test_dataset = CommonCorruptionDataset(datadir, dataset, corruption, test_transform, severity)
        else:
            train_dataset = CIFAR100(datadir, True, train_transform, download=True)
            test_dataset = CIFAR100(datadir, False, test_transform, download=True)
    elif dataset == 'ImageNet16-120':
        train_dataset = ImageNet16(datadir, True, train_transform, use_num_of_class_only=120, corruption=corruption)
        test_dataset  = ImageNet16(datadir, False, test_transform , use_num_of_class_only=120, corruption=corruption)
    elif dataset == 'ImageNet1k':
        train_dataset = H5Dataset(os.path.join(datadir, 'imagenet-train-256.h5'), transform=train_transform)
        test_dataset  = H5Dataset(os.path.join(datadir, 'imagenet-val-256.h5'),   transform=test_transform)            
    else:
        raise ValueError('There are no more cifars or imagenets.')

    if return_ds:
        return train_dataset, test_dataset

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader


##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################

def calculate_md5(fpath, chunk_size=1024 * 1024):
  md5 = hashlib.md5()
  with open(fpath, 'rb') as f:
    for chunk in iter(lambda: f.read(chunk_size), b''):
      md5.update(chunk)
  return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
  return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
  if not os.path.isfile(fpath): return False
  if md5 is None: return True
  else          : return check_md5(fpath, md5)


class ImageNet16(Dataset):
  # http://image-net.org/download-images
  # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
  # https://arxiv.org/pdf/1707.08819.pdf
  
    train_list = [
        ['train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'],
        ['train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'],
        ['train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'],
        ['train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'],
        ['train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'],
        ['train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'],
        ['train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'],
        ['train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'],
        ['train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'],
        ['train_data_batch_10','8f03f34ac4b42271a294f91bf480f29b'],
    ]
    valid_list = [
        ['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
    ]
    
    def __init__(self, root, train, transform, corruption=None, use_num_of_class_only=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        if not self._check_integrity(): 
            raise RuntimeError('Dataset not found or corrupted.')

        if self.train: downloaded_list = self.train_list
        else: downloaded_list = self.valid_list
        self.data, self.targets = [], []
        self.corruption = corruption
    
        # now load the picked numpy arrays
        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
        
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            self.targets.extend(entry['labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if use_num_of_class_only is not None:
            assert isinstance(use_num_of_class_only, int) and use_num_of_class_only > 0 and use_num_of_class_only < 1000, 'invalid use_num_of_class_only : {:}'.format(use_num_of_class_only)
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if 1 <= L <= use_num_of_class_only:
                    new_data.append(I)
                    new_targets.append(L)
            self.data = new_data
            self.targets = new_targets
    
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]-1
        if self.corruption is not None:
            img = corrupt(img, corruption_name=self.corruption)
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


    def __len__(self):
        return len(self.data)


    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.valid_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
        if not check_integrity(fpath, md5):
            return False
        return True

  