# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-12-24 06:26:23
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2022-03-26 21:59:41
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Spiking Neuron models
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config.base_config import args
pi = torch.tensor(math.pi)


class Linear(torch.autograd.Function):
    '''
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    '''
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = Linear.gamma * F.threshold(1.0 - torch.abs(inpt), 0, 0)
        return grad_input * sur_grad.float()


class Rectangle(torch.autograd.Function):
    '''
    Here we use the Rectangle surrogate gradient as was done
    in Yu et al. (2018).
    '''
    # beta = 1.0  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = (torch.abs(inpt) < 0.5).float()
        return grad_input * sur_grad


class PDF(torch.autograd.Function):

    alpha = 0.1
    beta = 0.1

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = PDF.alpha * torch.exp(-PDF.beta * torch.abs(inpt))
        return sur_grad * grad_input


class LIF(nn.Module):
    '''
        Forward Return: spikes in each time step
    '''

    def __init__(self, in_feature):
        super(LIF, self).__init__()
        self.in_feature = in_feature
        self.v_th = args.v_th
        self.v_decay = args.v_decay
        if args.sur_grad == 'linear':
            self.act_fun = Linear.apply
        elif args.sur_grad == 'rectangle':
            self.act_fun = Rectangle.apply
        elif args.sur_grad == 'pdf':
            self.act_fun = PDF.apply

    def reset_parameters(self, inpt):
        self.membrane = inpt

    def forward(self, inpt, step):
        if step == 0:   # reset
            self.reset_parameters(inpt)
        else:
            self.membrane = self.v_decay * self.membrane * (1. - self.spike) + inpt
        if args.sur_grad == 'linear':
            mem_thr = self.membrane / self.v_th - 1.0
        if args.sur_grad == 'rectangle':
            mem_thr = self.membrane - self.v_th
        self.spike = self.act_fun(mem_thr)
        return self.spike

    def extra_repr(self):
        return 'in_feature={}'.format(self.in_feature)


class A_LIF(nn.Module):
    # Shared adaptative threshold and Leaky in each layer
    def __init__(self, in_feature):
        super(A_LIF, self).__init__()
        self.in_feature = in_feature
        self.v_th = nn.Parameter(torch.tensor(args.v_th).float(), requires_grad=True)
        # self.v_th = torch.tensor(args.v_th)
        self.v_decay = nn.Parameter(torch.tensor(args.v_decay).float(), requires_grad=True)
        if args.sur_grad == 'linear':
            self.act_fun = Linear().apply
        elif args.sur_grad == 'rectangle':
            self.act_fun = Rectangle().apply
        elif args.sur_grad == 'pdf':
            self.act_fun = PDF().apply

    def reset_parameters(self, inpt):
        self.membrane = inpt

    def forward(self, inpt, step):
        if step == 0:   # reset
            self.reset_parameters(inpt)
        else:
            self.membrane = torch.sigmoid(self.v_decay) * self.membrane * (1. - self.spike) + inpt
            # self.membrane = self.v_decay * self.membrane * (1. - self.spike) + inpt

        # mem_thr = self.membrane / self.v_th - 1.0
        # mem_thr = self.membrane / torch.sigmoid(self.v_th) - 1.0

        if self.v_th > 0:
            mem_thr = self.membrane / (self.v_th + 1e-8) - 1.0
        elif self.v_th <= 0:
            mem_thr = 1.0 - self.membrane / (self.v_th + 1e-8)

        self.spike = self.act_fun(mem_thr)
        return self.spike.clone()

    def extra_repr(self):
        return 'in_feature={}'.format(self.in_feature)
