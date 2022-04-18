# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-12-24 03:37:45
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2022-03-22 20:44:12
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : functional function library
'''
import os
import math
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, log_file):
        self.logger = logging.getLogger('')
        # file handler
        handler = logging.FileHandler(filename=log_file, mode="w")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(message)s'))

        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)
        self.logger.info("Logger created at {}".format(log_file))

    def debug(self, strout):
        return self.logger.debug(strout)

    def info(self, strout):
        return self.logger.info(strout)


# like switch-case
def set_loss(var):
    return {
        'ce': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),
    }.get(var, 'error')


# like switch-case
def set_optimizer(var, model, lr):
    return {
        'sgd': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        'adam': optim.Adam(model.parameters(), lr=lr)
    }.get(var, 'error')


def accuracy(output, target, args):
    # output = output.view(output.size(0), args.output_size, 1)
    # vote_sum = torch.sum(output, dim=2)
    # pred = torch.argmax(vote_sum, dim=1)
    _, pred = output.cpu().max(1)
    correct = torch.eq(pred, target.long()).sum().item()
    return correct


def lr_scheduler(optimizer, epoch, lr_decay_epoch=30, lr_decay=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_decay
    return optimizer


def bernoulli(datum: torch.Tensor, time, dt: float = 1.0, **kwargs) -> torch.Tensor:
    max_prob = kwargs.get('max_prob', 1.0)
    assert 0 <= max_prob <= 1, 'Maximum firing probability must be in range [0, 1]'
    assert (datum >= 0).all(), 'Inputs must be non-negative'
    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    if time is not None:
        time = int(time / dt)
    # Normalize inputs and rescale (spike probability proportional to normalized intensity).
    if datum.max() > 1.0:
        datum /= datum.max()
    # Make spike data from Bernoulli sampling.
    if time is None:
        spikes = torch.bernoulli(max_prob * datum)
        spikes = spikes.view(*shape)
    else:
        spikes = torch.bernoulli(max_prob * datum.repeat([time, 1]))
        spikes = spikes.view(time, *shape)
    return spikes.float()


def fixed_uniform(datum: torch.Tensor, time, dt: float = 1.0) -> torch.Tensor:
    assert (datum >= 0).all(), 'Inputs must be non-negative'
    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    if time is not None:
        time = int(time / dt)
    # Normalize inputs and rescale (spike probability proportional to normalized intensity).
    if datum.max() > 1.0:
        datum /= datum.max()
    datum_idx = torch.where(datum > 0)
    # fire_num = (datum[datum_idx] * time).round().long()
    interval = (1 / datum[datum_idx]).floor()
    fire_arr = interval.repeat(time, 1).T
    fire_time = torch.cumsum(fire_arr, dim=1).long() - 1
    fire_time[fire_time < 0] = 0
    # fire_time[fire_time >= time] = -1
    # print(fire_time[0, :])
    new_idx = datum_idx[0].unsqueeze(1).expand_as(fire_time)
    fire_time = fire_time.reshape(-1)
    new_idx = new_idx.reshape(-1)
    # print(fire_time[0:time])
    # print(new_idx[0:time])
    idx = torch.where(fire_time < time)
    x_idx = fire_time[idx]
    # print(x_idx[0:time])
    y_idx = new_idx[idx]
    # print(y_idx[0:time])
    spikes = torch.zeros(time, datum.size(0))
    spikes[x_idx, y_idx] = 1.0
    # for i in range(int(interval.size(0))):
    #     # fire_arr = interval[i] * torch.ones(int(fire_num[i]), 1)
    #     # fire_time = torch.cumsum(fire_arr, dim=0).long() - 1
    #     # fire_time[fire_time >= time] = 0
    #     # fire_time[fire_time < 0] = 0
    #     fire_time = torch.arange(0, time, interval[i]).long()
    #     spikes[fire_time, datum_idx[0][i]] = 1.0
    spikes = spikes.view(time, *shape)
    return spikes.float()


def plot_CM(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='sans-serif', size='15')   # 设置字体样式、大小
    # plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'SimHei', 'Lucida Grande', 'Verdana']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.figure()
    plt.rcParams['figure.dpi'] = 200  # 分辨率

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    plt.title('Confusion matrix')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=list(range(len(classes))), yticklabels=list(range(len(classes))),
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.05)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # plt.show()
    plt.savefig('./CM.png', dpi=600)
