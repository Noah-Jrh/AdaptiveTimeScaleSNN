# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-12-24 02:48:37
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2022-03-28 17:48:13
   @CopyRight (c): 2020 NCRC, SCU. All rights reserved.
   @Description  : Deep SNN
'''
import time
import tqdm
import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
from config.base_config import args
import utils.function as F
from utils.data import dataloader
from modules.model import SpkConvNet
# from modules.neuron import A_LIF, LIF
from torch.utils.tensorboard import SummaryWriter

# Set data
dataset_train, dataset_test = dataloader()
# Init Network
# model = torch.nn.DataParallel(SpkConvNet()).to(args.device)
model = SpkConvNet().to(args.device)
if args.train:
    log_name = os.path.join(args.log_path, 'train.log')
else:
    log_name = os.path.join(args.logs_path, 'test.log')
# Set Logger
logger = F.Logger(log_name)
argsDict = args.__dict__
logger.info('>>>> Hyper-parameters')
for key, value in sorted(argsDict.items()):
    logger.info(' --' + str(key) + ': ' + str(value))
logger.info('>>>> Learner Structure:')
# logger.info(model.net)
logger.info(model)
logger.info('>>>> Learner Parameters:')
for name, params in model.named_parameters():
    logger.info(' --' + str(name) + ': ' + str(params.shape))


def loss_calc(output, label, loss_fun):
    batch = output.size(0)
    if args.loss_fun == 'ce':
        loss = loss_fun(output, label.long())
    elif args.loss_fun == 'mse':
        label = torch.zeros(batch, args.output_size).scatter_(1, label.view(-1, 1), 1)
        loss = loss_fun(output, label)
    else:
        raise NameError("Loss {} not recognized".format(args.loss_fun))
    return loss


def runTrain(epoch, pbar, model, dataset_train, evaluator, optimizer):
    tot_loss = 0
    tot_correct = 0
    tot_num = 0
    for batch_idx, (samples, labels) in enumerate(dataset_train):
        model.train()
        model.zero_grad()
        opt_train = []
        # encoding layer
        if args.encoding == "poison":
            input = F.bernoulli(samples, args.T).to(args.device)
        elif args.encoding == "fixed":
            input = F.fixed_uniform(samples, args.T).to(args.device)
        elif args.encoding == "conv":
            samples = samples.to(args.device)
            input = model.net[1](model.net[0](samples))
            # input = model.net[0](samples)
        else:
            input = samples.float().to(args.device)
        # print(input.size())
        # forward layer
        for step in range(args.T):
            if args.encoding == 'poison' or args.encoding == 'fixed':
                x = input[step, :]
            elif args.encoding == "dvs":
                x = input[..., step]
            elif args.encoding == "conv":
                x = input
            else:
                x = input[..., step]
            y = model(x, step)
            opt_train.append(y)
        # Voltage output: Suit Voltage-based loss (CE)
        if args.loss_fun == 'ce':
            output = np.sum(opt_train, 0)
        # Spike output: Suit Spike-based loss (MSE)
        elif args.loss_fun == 'mse':
            output = np.sum(opt_train, 0) / args.T
        optimizer.zero_grad()
        loss = loss_calc(output.cpu(), labels, evaluator)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        tot_correct += F.accuracy(output, labels, args)
        tot_num += samples.shape[0]
        train_loss = tot_loss / tot_num
        train_acc = tot_correct / tot_num * 100
        pbar.set_postfix({"LOSS": train_loss, "ACC": train_acc})
    return train_loss, train_acc


def runTest(dataset_test, model, evaluator):
    model.eval()
    with torch.no_grad():
        tot_correct = 0
        tot_num = 0
        tot_loss = 0
        for batch_idx, (samples, labels) in enumerate(dataset_test):
            model.eval()
            opt_train = []
            if args.encoding == "poison":
                input = F.bernoulli(samples, args.T).to(args.device)
            elif args.encoding == "fixed":
                input = F.fixed_uniform(samples, args.T).to(args.device)
            elif args.encoding == "conv":
                samples = samples.to(args.device)
                input = model.net[1](model.net[0](samples))
                # input = model.net[0](samples)
            else:
                input = samples.float().to(args.device)
            # forward layer
            for step in range(args.T):
                if args.encoding == 'poison' or args.encoding == 'fixed':
                    x = input[step, :]
                elif args.encoding == "dvs":
                    x = input[..., step]
                elif args.encoding == "conv":
                    x = input
                else:
                    x = input[:, :, step]
                y = model(x, step)
                opt_train.append(y)
            # Voltage output: Suit Voltage-based loss (CE)
            if args.loss_fun == 'ce':
                output = np.sum(opt_train, 0)
            # Spike output: Suit Spike-based loss (MSE)
            elif args.loss_fun == 'mse':
                output = np.sum(opt_train, 0) / args.T
            loss = loss_calc(output.cpu(), labels, evaluator)
            _, pred = output.cpu().max(1)
            tot_loss += loss.item()
            tot_correct += F.accuracy(output, labels, args)
            tot_num += samples.shape[0]
        test_loss = tot_loss / tot_num
        test_acc = tot_correct / tot_num * 100
    return test_loss, test_acc


# def confusion_matrix(output, target, args):
#     cm = torch.zeros(args.n_way, args.n_way)
#     output = output.view(output.size(0), args.n_way, args.output_size // args.n_way)
#     vote_sum = torch.sum(output, dim=2)
#     pred = torch.argmax(vote_sum, dim=1)
#     for p, t in zip(pred, target):
#         cm[p, t] += 1
#     return cm.numpy().T


# BASE_ACC = {'MNIST': 0.98, 'CIFAR10': 0.84, 'NMNIST': 0.98, 'TIDIGITS': 0.95, 'RWCP': 0.98}

# MAX_ACC = {'MNIST': 99.33, 'CIFAR10': 90.82, 'NMNIST': 98.61, 'TIDIGITS': 96.69, 'RWCP': 100.00}
# BEST_EPOCH = {'BPTT_SCN_3v1_MNIST': 'Epoch_93.pth',
#               'BPTT_SCN_4v1_MNIST': 'Epoch_64.pth',
#               'BPTT_SCN_8v1_CIFAR10': 'Epoch_92.pth',
#               'BPTT_SCN_3v1_NMNIST': 'Epoch_61.pth',
#               'BPTT_SNN_2v1_RWCP': 'Epoch_10.pth',
#               'BPTT_SNN_2v1_TIDIGITS': 'Epoch_17.pth'}


def main():
    if args.train:
        best_acc = 0
        train_trace, val_trace = dict(), dict()
        train_trace['acc'], train_trace['loss'] = [], []
        val_trace['acc'], val_trace['loss'] = [], []
        writer = SummaryWriter(args.log_path)
        evaluator = F.set_loss(args.loss_fun)
        optimizer = F.set_optimizer(args.optimizer, model, args.lr)
        if args.scheduler == 'CosineAL':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nEpoch / 16)
        pbar = tqdm.tqdm(range(args.nEpoch))
        start_time = time.time()
        for epoch in pbar:
            train_loss, train_acc = runTrain(epoch, pbar, model, dataset_train, evaluator, optimizer)
            logger.info('Epoch [%d/%d], Loss: %.5f, Acc: %.2f, Time elasped: %.2f'
                        % (epoch + 1, args.nEpoch, train_loss, train_acc, time.time() - start_time))
            start_time = time.time()
            if args.scheduler == 'CosineAL':
                scheduler.step()
            val_loss, val_acc = runTest(dataset_test, model, evaluator)
            logger.info('Val_Loss: % .5f, Val_Acc: % .2f' % (val_loss, val_acc))

            # Tensorboard Record
            writer.add_scalars('loss', {'val': val_loss, 'train': train_loss}, epoch + 1)
            writer.add_scalars('acc', {'val': val_acc, 'train': train_acc}, epoch + 1)

            # Checkpoints Record
            train_trace['acc'].append(train_acc)
            train_trace['loss'].append(train_loss)
            val_trace['acc'].append(val_acc)
            val_trace['loss'].append(val_loss)

            if (val_acc > best_acc):
                best_acc = val_acc
                best_epoch = epoch
                logger.info('Saving model..  with acc {0} in the epoch {1}'.format(best_acc, epoch))
                state = {
                    'best_acc': best_acc,
                    'best_epoch': best_epoch,
                    'best_net': model.state_dict(),
                    'config': args
                }
                torch.save(state, os.path.join(args.ckpt_path, 'best_ckpt.pth'))

            if (epoch + 1) == args.nEpoch:
                logger.info('Best acc is {} in epoch {}. The relate checkpoint path: {}'.format(best_acc, best_epoch, os.path.join('Best.pth')))
                state = {
                    'traces': {'train': train_trace, 'val': val_trace},
                }
                torch.save(state, os.path.join(args.ckpt_path, 'trace.pth'))
    else:
        checkpoint = torch.load(os.path.join(args.ckpt_path))
        logger.info('Best acc is {} in epoch {}.:'.format(checkpoint['best_acc'], checkpoint['best_epoch']))
        model.load_state_dict(checkpoint['best_net'])
        val_loss, val_acc = runTest(dataset_test, model, evaluator)
        logger.info('Val_Loss: % .5f, Val_Acc: % .2f' % (val_loss, val_acc))


if __name__ == '__main__':
    main()
