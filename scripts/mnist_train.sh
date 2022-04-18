# coding=utf-8
### 
   # @Author       : Noah
   # @Version      : v1.0.0
   # @Date         : 2020-12-24 10:30:28
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2022-03-26 17:43:08
   # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   # @Description  : Please add descriptioon
### 
# --scheduler 'CosineAL' \
# For MNIST 
# A_LIF conv_coding SCN_4v1 2020-12-27-08_08_25
# A_LIF rate_coding SCN_3v1 2020-12-27-08_08_25
export CUDA_VISIBLE_DEVICES=1
python ../main.py \
   --model 'BPTT' \
   --structure 'SCN_4v1' \
   --dataSet 'MNIST' \
   --dataDir '/opt/data/durian/Benchmark' \
   --resDir '../results' \
   --encoding 'fixed' \
   --nEpoch 240 \
   --batchSize 32 \
   --v_th 1.0 \
   --v_decay -0.8 \
   --T 8 \
   --sur_grad 'linear' \
   --loss_fun 'ce' \
   --optimizer 'adam' \
   --scheduler 'CosineAL' \
   --lr 0.0001 \
   --train