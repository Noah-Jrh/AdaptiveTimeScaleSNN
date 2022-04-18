# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2022-03-22 10:32:09
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2022-03-29 16:07:36
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Please add descriptioon
'''
import os
import time
import torch
import random
import argparse
import numpy as np

'''checkpoints
MNIST LIF 2022-03-23-11_35_04
'''

parser = argparse.ArgumentParser(description='PLIF_SNN')
parser.add_argument("--model", type=str, default='BPTT')
parser.add_argument("--structure", type=str, default='SCN_8v1')
parser.add_argument("--dataSet", type=str, default='CIFAR10',
                    choices=['MNIST', 'CIFAR10', 'NMNIST', 'TIDIGITS', 'RWCP'])
parser.add_argument("--encoding", choices=['poison', 'fixed', 'conv', 'dvs', None], default=None)
parser.add_argument("--checkpoints", type=str, default='2022-03-24-23_11_39')
parser.add_argument("--dataDir", type=str, default='/opt/data/durian/Benchmark/')
parser.add_argument("--resDir", type=str, default='./results')
parser.add_argument("--nEpoch", type=int, default=240)
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--seed", type=int, default=None, help="Random seed")

parser.add_argument("--v_th", type=float, default=0.5)
parser.add_argument("--v_decay", type=float, default=10)
parser.add_argument("--T", type=int, default=8)
parser.add_argument("--dt", type=int, default=1, help="ms")

parser.add_argument("--sur_grad", choices=['linear', 'rectangle', 'pdf'], default='linear')
parser.add_argument('--loss_fun', choices=['ce', 'mse'], default='ce',
                    help='Loss Function: ce(CrossEntropyLoss), mse(MSELoss)')
parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam',
                    help='Optimizer: SGD, Adam')
parser.add_argument('--scheduler', choices=['CosineAL'], default=None)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument("--learner_bias", dest="learner_bias", action="store_true")
parser.add_argument("--train", dest="train", action="store_true")
parser.set_defaults(learner_bias=False, train=False)
args = parser.parse_known_args()[0]
# if len(unparsed) != 0:
#     raise NameError("Argument {} not recognized".format(unparsed))

# Set Input and Output size
if args.dataSet == 'MNIST':
    args.img_size = [28, 28, 1]
    args.output_size = 10
elif args.dataSet == 'CIFAR10':
    args.img_size = [32, 32, 3]
    args.output_size = 10
elif args.dataSet == 'NMNIST':
    args.img_size = [34, 34, 2]
    args.output_size = 10
elif args.dataSet == 'RWCP':
    args.input_size = 576
    args.output_size = 10
elif args.dataSet == 'TIDIGITS':
    args.input_size = 576
    args.output_size = 11
else:
    raise NameError("Argument {} not recognized".format(args.dataSet))


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# Set path
if args.train:
    flag = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
else:
    flag = args.checkpoints
names = [args.model, args.structure, args.dataSet]
model_name = '_'.join([str(x) for x in names])
data_path = os.path.join(args.dataDir, args.dataSet)
log_path = os.path.join(args.resDir, model_name, flag, "log")
ckpt_path = os.path.join(args.resDir, model_name, flag, "checkpoints")
for path in [data_path, log_path, ckpt_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

args.data_path = data_path
args.log_path = log_path
args.ckpt_path = ckpt_path

# Set seed
if args.seed is None:
    args.seed = random.randint(0, 1e3)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


''' Configuration instructions
    CONV: nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding) in_planes from previous layer
    AP: nn.AvgPool2d(kernel_size, stride, padding)
    FC: nn.Linear(in_features, out_features) in_planes from previous layer
    A_LIF : nm.A_LIF()
    Spiking Convolutional Neural Network (SCN)
        SCN_5_32: 4 Conv-Avgpool layers (32 filters + Nonoverlap AvgPool) + 1-layer Fully Connection
'''
cfg = {
    # For TIDIGITS + LIF
    'SNN_2': [('linear', [256]), ('lif', []), ('dp', [0.5]), ('output', [args.output_size])],
    'SNN_3': [('linear', [500]), ('lif', []), ('linear', [500]), ('lif', []),
              ('output', [args.output_size])],
    # For TIDIGITS + A_LIF
    'SNN_2v1': [('linear', [256]), ('a_lif', []), ('dp', [0.5]), ('output', [args.output_size])],
    'SNN_3v1': [('linear', [500]), ('a_lif', []), ('dp', [0.5]), ('linear', [500]), ('a_lif', []),
                ('dp', [0.5]), ('output', [args.output_size])],

    # For MNIST/NMNIST + LIF 3:rate coding  4:conv coding
    # 'SCN_4': [('conv2d', [128, 3, 1, 1]), ('bn2', [1e-5, 0.1]), ('lif', []), ('ap', [2, 2, 0]),
    #           ('conv2d', [128, 3, 1, 1]), ('lif', []), ('ap', [2, 2, 0]),
    #           ('flatten', []), ('linear', [128 * 4 * 4]), ('lif', []),
    #           ('output', [args.output_size])],

    'SCN_4': [('conv2d', [128, 3, 1, 1]), ('bn2', [1e-5, 0.1]), ('lif', []), ('ap', [2, 2, 0]),
              ('conv2d', [128, 3, 1, 1]), ('bn2', [1e-5, 0.1]), ('lif', []), ('ap', [2, 2, 0]),
              ('flatten', []), ('dp', [0.5]), ('linear', [128 * 4 * 4]), ('lif', []),
              ('dp', [0.5]), ('output', [args.output_size])],

    # For MNIST/NMNIST + A_LIF 3v1:rate coding  4v1:conv coding
    # 'SCN_4v1': [('conv2d', [128, 3, 1, 1]), ('bn2', [1e-5, 0.1]), ('a_lif', []), ('ap', [2, 2, 0]),
    #             ('conv2d', [128, 3, 1, 1]), ('bn2', [1e-5, 0.1]), ('a_lif', []), ('ap', [2, 2, 0]),
    #             ('flatten', []), ('dp', [0.5]), ('linear', [128 * 4 * 4]), ('a_lif', []),
    #             ('dp', [0.5]), ('output', [args.output_size])],

    'SCN_4v1': [('conv2d', [128, 3, 1, 1]), ('bn2', [1e-5, 0.1]), ('a_lif', []), ('ap', [2, 2, 0]),
                ('conv2d', [128, 3, 1, 1]), ('a_lif', []), ('ap', [2, 2, 0]),
                ('flatten', []), ('linear', [128 * 4 * 4]), ('a_lif', []),
                ('output', [args.output_size])],

    # For CIFAR10 + LIF
    'SCN_8': [('conv2d', [128, 3, 1, 1]), ('bn2', [1e-5, 0.1]), ('lif', []), ('dp', [0.5]),
              ('conv2d', [256, 3, 1, 1]), ('lif', []), ('ap', [2, 2, 0]),
              ('dp', [0.5]), ('conv2d', [512, 3, 1, 1]), ('lif', []), ('ap', [2, 2, 0]),
              ('dp', [0.5]), ('conv2d', [1024, 3, 1, 1]), ('lif', []),
              ('dp', [0.5]), ('conv2d', [512, 3, 1, 1]), ('lif', []),
              ('flatten', []), ('dp', [0.5]), ('linear', [1024]), ('lif', []),
              ('dp', [0.5]), ('linear', [512]), ('lif', []),
              ('dp', [0.5]), ('output', [args.output_size])],

    # For CIFAR10 + A_LIF
    'SCN_8v1': [('conv2d', [128, 3, 1, 1]), ('bn2', [1e-5, 0.1]), ('a_lif', []), ('dp', [0.5]),
                ('conv2d', [256, 3, 1, 1]), ('a_lif', []), ('ap', [2, 2, 0]),
                ('dp', [0.5]), ('conv2d', [512, 3, 1, 1]), ('a_lif', []), ('ap', [2, 2, 0]),
                ('dp', [0.5]), ('conv2d', [1024, 3, 1, 1]), ('a_lif', []),
                ('dp', [0.5]), ('conv2d', [512, 3, 1, 1]), ('a_lif', []),
                ('flatten', []), ('dp', [0.5]), ('linear', [1024]), ('a_lif', []),
                ('dp', [0.5]), ('linear', [512]), ('a_lif', []),
                ('dp', [0.5]), ('output', [args.output_size])],
}
