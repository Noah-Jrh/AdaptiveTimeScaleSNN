# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-12-24 06:11:53
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2022-03-29 21:22:32
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Please add descriptioon
'''
# from config.base_config import cfg
# import math
# import torch
# import numpy as np
import torch.nn as nn
# import torch.nn.functional as F
from functools import reduce
import modules.neuron as nm


class SpkConvNet(nn.Module):
    '''
        Spiking Neural Network and Convolutional Neural Network
    '''

    def __init__(self, args, cfg):
        super(SpkConvNet, self).__init__()
        self.args = args
        self.net = self._make_layers(cfg[args.structure])

    def _make_layers(self, cfg):
        layers = []
        if self.args.structure[0:3] == 'SNN':
            in_features = self.args.input_size
        else:
            in_channels = self.args.img_size[2]  # decided by the dataSet
            feature_map = [self.args.img_size[0], self.args.img_size[1]]
            in_features = reduce(lambda x, y: x * y, self.args.img_size)   # 列表中所有元素的乘积
        for i, (name, param) in enumerate(cfg):
            if name == 'conv2d':
                layers += [nn.Conv2d(in_channels, param[0], kernel_size=param[1], stride=param[2], padding=param[3], bias=self.args.learner_bias)]
                feature_map = [(x - param[1] + 2 * param[3]) // param[2] + 1 for x in feature_map]  # W = (W - K + 2P) / S + 1
                in_channels = param[0]
            elif name == 'dp':
                layers += [nn.Dropout(param[0])]
            elif name == 'bn2':
                layers += [nn.BatchNorm2d(in_channels, eps=param[0], momentum=param[1])]
            elif name == 'a_lif':
                if isinstance(layers[-1], (nn.BatchNorm2d, nn.Conv2d, nn.Dropout)):
                    layers += [nm.A_LIF([in_channels] + feature_map)]
                elif isinstance(layers[-1], (nn.Linear)):
                    layers += [nm.A_LIF(in_features)]
            elif name == 'lif':
                if isinstance(layers[-1], (nn.BatchNorm2d, nn.Conv2d, nn.Dropout)):
                    layers += [nm.LIF([in_channels] + feature_map)]
                elif isinstance(layers[-1], (nn.Linear)):
                    layers += [nm.LIF(in_features)]
            elif name == 'ap':
                layers += [nn.AvgPool2d(kernel_size=param[0], stride=param[1], padding=param[2])]
                feature_map = [(x - param[0] + 2 * param[2]) // param[1] + 1 for x in feature_map]  # W = (W - K + 2P) / S + 1
            elif name == 'flatten':
                in_features = in_channels * reduce(lambda x, y: x * y, feature_map)   # 列表中所有元素的乘积
            elif name == 'linear':
                layers += [nn.Linear(in_features, param[0], bias=self.args.learner_bias)]
                in_features = param[0]
            elif name == 'output':
                layers += [nn.Linear(in_features, param[0], bias=self.args.learner_bias)]
                if self.args.loss_fun == 'ce':
                    break
                elif self.args.loss_fun == 'mse':
                    layers += [nm.LIF(param)]
                else:
                    raise NameError("Loss Functions {} not recognized".format(name))
            else:
                raise NameError("Components {} not recognized".format(name))
        # nn.ModuleDict({'features': nn.Sequential(*layers)})
        net = nn.Sequential(*layers)
        return net

    def forward(self, x, step):
        batch = x.size(0)
        start = 0
        if self.args.encoding == "conv":
            start = 2
        for i in range(start, len(self.net)):
            if isinstance(self.net[i], (nn.Conv2d, nn.AvgPool2d, nn.BatchNorm2d, nn.Dropout)):
                # print(i, self.net[i])
                x = self.net[i](x)
            elif isinstance(self.net[i], (nn.Linear)):
                # print(i, self.net[i])
                x = x.view(batch, -1)
                x = self.net[i](x)
            elif isinstance(self.net[i], (nm.LIF, nm.A_LIF)):
                # print(i, self.net[i])
                x = self.net[i](x, step)
        return x
