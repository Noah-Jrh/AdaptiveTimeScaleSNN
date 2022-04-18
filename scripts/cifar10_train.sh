# coding=utf-8
### 
   # @Author       : Noah
   # @Version      : v1.0.0
   # @Date         : 2020-12-24 10:30:28
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2022-03-23 11:27:04
   # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   # @Description  : Please add descriptioon
### 

# export CUDA_VISIBLE_DEVICES=2
# python ../main.py \
#    --model 'BPTT' \
#    --structure 'SCN_8' \
#    --dataSet 'CIFAR10' \
#    --dataDir './Data' \
#    --resDir '../results' \
#    --nEpoch 100 \
#    --batchSize 50 \
#    --img_size 32 32 3 \
#    --input_size 1024 \
#    --output_size 10 \
#    --v_th 0.5 \
#    --v_decay 1.0 \
#    --T 12 \
#    --step 12 12 \
#    --sur_grad 'rectangle' \
#    --loss_fun 'mse' \
#    --optimizer 'sgd' \
#    --lr 0.1 \
#    --train \
#    --learner_bias

# seed 522
# conv_encoding
export CUDA_VISIBLE_DEVICES=1
python ../main.py \
   --model 'BPTT' \
   --structure 'SCN_8' \
   --dataSet 'CIFAR10' \
   --dataDir '/DATA/runhao/Data/Benchmark' \
   --resDir '../results' \
   --encoding 'conv' \
   --nEpoch 100 \
   --batchSize 64 \
   --v_th 0.5 \
   --v_decay 0.5 \
   --T 8 \
   --sur_grad 'linear' \
   --loss_fun 'ce' \
   --optimizer 'adam' \
   --lr 0.0001 \
   --train

# # rate_encoding
# export CUDA_VISIBLE_DEVICES=2
# python ../main.py \
#    --model 'BPTT' \
#    --structure 'SCN_8v1' \
#    --dataSet 'CIFAR10' \
#    --dataDir './Data' \
#    --resDir '../results' \
#    --encoding 'rate' \
#    --nEpoch 100 \
#    --batchSize 50 \
#    --seed 49 \
#    --img_size 32 32 3 \
#    --input_size 1024 \
#    --output_size 10 \
#    --v_th 0.5 \
#    --v_decay 1.0 \
#    --T 12 \
#    --step 12 12 \
#    --sur_grad 'linear' \
#    --loss_fun 'ce' \
#    --optimizer 'adam' \
#    --lr 0.0001 \
#    --train