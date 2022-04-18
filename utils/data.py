# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-12-24 05:11:06
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2022-03-28 17:43:24
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : DataLoader, Preprocess
'''
import os
# import random
import numpy as np
import scipy.io as io
# from PIL import Image
import pathlib
# import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from config.base_config import args
# from aermanager.dataset_generator import gen_dataset_from_folders
# from aermanager.datasets import FramesDataset, SpikeTrainDataset
# from aermanager.parsers import parse_nmnist


def dataloader():
    if args.dataSet == 'MNIST':
        dataset = load_mnist(args.data_path)
    elif args.dataSet == 'CIFAR10':
        dataset = load_cifar10(args.data_path)
    elif args.dataSet == 'NMNIST':
        dataset = load_nmnist(args.data_path)
    elif args.dataSet == 'TIDIGITS':
        dataset = load_tidigits(args.data_path)
    elif args.dataSet == 'RWCP':
        dataset = load_rwcp(args.data_path)
    else:
        raise (ValueError('Unsupported dataset'))
    return dataset


def load_mnist(data_path: str):
    """ Load MNIST
    :param data_path:
    :param args:
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    if args.encoding == 'poison' or args.encoding == 'fixed':
        train_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=train_transforms)
    # train_nums = train_dataset.data.shape[0]
    # args.img_size = [train_dataset.data.shape[1], train_dataset.data.shape[2], 1]
    # args.input_size = train_dataset.data.shape[1] * train_dataset.data.shape[2]
    train_loader = data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4)

    test_set = datasets.MNIST(root=data_path, train=False, download=True,
                              transform=test_transforms)
    # test_nums = test_set.data.shape[0]
    test_loader = data.DataLoader(test_set, batch_size=args.batchSize, shuffle=False, num_workers=4)
    return train_loader, test_loader


def load_cifar10(data_path: str):
    """ Load CIFAR10
    :param data_path:
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    if args.encoding == 'poison' or args.encoding == 'fixed':
        train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=6),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.ToTensor()])
    else:
        train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=6),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                              ])

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transforms)

    # train_nums = train_dataset.data.shape[0]
    # args.img_size = [train_dataset.data.shape[1], train_dataset.data.shape[2], train_dataset.data.shape[3]]
    # args.input_size = train_dataset.data.shape[1] * train_dataset.data.shape[2]
    train_loader = data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=0)

    test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transforms)
    # test_nums = test_set.data.shape[0]
    test_loader = data.DataLoader(test_set, batch_size=args.batchSize, shuffle=False, num_workers=0)
    return train_loader, test_loader


def find_classes(root_dir):
    # 统计样本数量
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        dirs.sort()
        for f in files:
            if f.endswith("png") or f.endswith("mat"):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    # 统计类别数量
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


def Event2Frame(sample, img_size, dt, T):
    dt = int(dt * 1e3)
    events = io.loadmat(sample, squeeze_me=True, struct_as_record=False)
    frame = np.zeros([2, img_size[0], img_size[1], T], dtype=int)  # frame
    for j in range(0, int(T * dt), int(dt)):    # tr ms 的帧
        idx_n = (events['TD'].ts >= j) & (events['TD'].ts < j + dt) & (events['TD'].p == 1)
        idx_p = (events['TD'].ts >= j) & (events['TD'].ts < j + dt) & (events['TD'].p == 2)
        frame[0, events['TD'].y[idx_n] - 1, events['TD'].x[idx_n] - 1, int(j / dt)] = 1.0
        frame[1, events['TD'].y[idx_p] - 1, events['TD'].x[idx_p] - 1, int(j / dt)] = 1.0
    #     im = Image.fromarray((frame[int(j / self.tr), :] * 255).astype(np.uint8), "L")  # numpy 转 image类
    #     im.save(os.path.join('./', str(j / self.tr) + '.png'))
    return frame


class NMNIST(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        if train:
            self.root = os.path.join(root + '/Train')
        else:
            self.root = os.path.join(root + '/Test')
        self.transform = transform
        self.target_transform = target_transform

        # 检查数据集路径
        if not os.path.exists(self.root):
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.all_items = find_classes(self.root)
        self.idx_classes = index_classes(self.all_items)
        self.samples = []
        for idx in range(len(self.all_items)):
            filename = self.all_items[idx][0]
            classname = self.all_items[idx][1]
            filepath = self.all_items[idx][2]
            events = os.path.join(str(filepath), str(filename))
            frame = Event2Frame(events, args.img_size, args.dt, args.T)
            target = self.idx_classes[classname]
            self.samples.append((frame, target))

    def __getitem__(self, index):
        img, target = self.samples[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.all_items)


def load_nmnist(data_path: str):
    """ Load NMNIST
    :param data_path:
    :param args:
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    train_dataset = NMNIST(root=data_path, train=True, download=False)
    train_loader = data.DataLoader(train_dataset, shuffle=True,
                                   num_workers=4, batch_size=args.batchSize)
    test_dataset = NMNIST(root=data_path, train=False, download=False)
    test_loader = data.DataLoader(test_dataset, shuffle=False,
                                  num_workers=4, batch_size=args.batchSize)
    return train_loader, test_loader


class TIDIGITS(data.Dataset):
    '''
    '''

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        if train:
            self.root = os.path.join(root + '/Train/')
            sample_name = 'ptnTrain'
            label_name = 'train_labels'
        else:
            self.root = os.path.join(root + '/Test')
            sample_name = 'ptnTest'
            label_name = 'test_labels'
        self.transform = transform
        self.target_transform = target_transform
        # 检查数据集路径
        if not os.path.exists(self.root):
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')
        path = pathlib.Path(self.root)
        evts = io.loadmat(path / 'samples.mat', squeeze_me=True, struct_as_record=False)[sample_name]
        labels = io.loadmat(path / 'labels.mat', squeeze_me=True, struct_as_record=False)[label_name]
        # args.T = 101
        # args.step = [args.T, args.T]
        self.samples = []
        # evts_vec = np.zeros([576, args.T], dtype=int)
        # find maxT
        # max_ts = []
        for i in range(evts.size):
            evts_vec = np.zeros([576, 101], dtype=int)
            evts[i][:, 0] = evts[i][:, 0] - 1
            evts[i][:, 1] = np.floor(evts[i][:, 1] * 1000) - 1
            evts[i] = evts[i].astype(np.int32)
            # max_ts.append(evts[i][:, 1].max())
            evts_vec[evts[i][:, 0], evts[i][:, 1]] = 1
            self.samples.append((evts_vec, labels[i] - 1))
        # print(max(max_ts))
        if train:
            print("Load TIDIGITS ptnTrain finish")
        else:
            print("Load TIDIGITS ptnTest finish")

    def __getitem__(self, index):
        img, target = self.samples[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.samples)


class RWCP(data.Dataset):
    '''
    '''

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        if train:
            self.root = os.path.join(root + '/Train/')
            sample_name = 'ptnTrain'
            label_name = 'train_labels'
        else:
            self.root = os.path.join(root + '/Test')
            sample_name = 'ptnTest'
            label_name = 'test_labels'
        self.transform = transform
        self.target_transform = target_transform
        # 检查数据集路径
        if not os.path.exists(self.root):
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')
        path = pathlib.Path(self.root)
        evts = io.loadmat(path / 'samples.mat', squeeze_me=True, struct_as_record=False)[sample_name]
        labels = io.loadmat(path / 'labels.mat', squeeze_me=True, struct_as_record=False)[label_name]
        # args.T = 101
        # args.step = [args.T, args.T]
        self.samples = []
        # evts_vec = np.zeros([576, args.T], dtype=int)
        # find maxT
        # max_ts = []
        for i in range(evts.size):
            evts_vec = np.zeros([576, 255], dtype=int)
            evts[i][:, 0] = evts[i][:, 0] - 1
            evts[i][:, 1] = np.floor(evts[i][:, 1] * 1000) - 1
            evts[i] = evts[i].astype(np.int32)
            # max_ts.append(evts[i][:, 1].max())
            evts_vec[evts[i][:, 0], evts[i][:, 1]] = 1
            self.samples.append((evts_vec, labels[i] - 1))
        # print(max(max_ts))
        # max_ts.sort(reverse=True)
        if train:
            print("Load RWCP ptnTrain finish")
        else:
            print("Load RWCP ptnTest finish")

    def __getitem__(self, index):
        img, target = self.samples[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.samples)


def load_tidigits(data_path: str):
    """ Load TIDIGITS
    :param data_path:
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    train_dataset = TIDIGITS(root=data_path, train=True, download=False)
    train_loader = data.DataLoader(train_dataset, shuffle=True,
                                   num_workers=4, batch_size=args.batchSize)
    test_dataset = TIDIGITS(root=data_path, train=False, download=False)
    test_loader = data.DataLoader(test_dataset, shuffle=True,
                                  num_workers=4, batch_size=args.batchSize)
    return train_loader, test_loader


def load_rwcp(data_path: str):
    """ Load TIDIGITS
    :param data_path:
    :param args:
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    train_dataset = RWCP(root=data_path, train=True, download=False)
    train_loader = data.DataLoader(train_dataset, shuffle=True,
                                   num_workers=4, batch_size=args.batchSize)
    test_dataset = RWCP(root=data_path, train=False, download=False)
    test_loader = data.DataLoader(test_dataset, shuffle=True,
                                  num_workers=4, batch_size=args.batchSize)
    return train_loader, test_loader
