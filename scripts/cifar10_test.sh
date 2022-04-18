# coding=utf-8
### 
   # @Author       : Noah
   # @Version      : v1.0.0
   # @Date         : 2020-12-24 10:30:28
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2021-01-01 11:56:02
   # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   # @Description  : Please add descriptioon
### 

# seed 522
# For CIFAR10
# A_LIF conv_coding SCN_8v1 2020-12-27-08_47_58
# A_LIF rate_coding SCN_8v1 2020-12-31-03_01_24
export CUDA_VISIBLE_DEVICES=2
python ../main.py \
   --model 'BPTT' \
   --structure 'SCN_8v1' \
   --dataSet 'CIFAR10' \
   --dataDir './Data' \
   --resDir '../results' \
   --checkpoints '2020-12-27-08_47_58' \
   --encoding 'conv' \
   --nEpoch 100 \
   --batchSize 50 \
   --seed 49 \
   --img_size 32 32 3 \
   --input_size 1024 \
   --output_size 10 \
   --v_th 0.5 \
   --v_decay 1.0 \
   --T 12 \
   --step 12 12 \
   --sur_grad 'linear' \
   --loss_fun 'ce' \
   --optimizer 'adam' \
   --lr 0.0001