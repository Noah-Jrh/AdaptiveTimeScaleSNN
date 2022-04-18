# coding=utf-8
### 
   # @Author       : Noah
   # @Version      : v1.0.0
   # @Date         : 2020-12-24 10:30:28
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2022-03-26 17:04:08
   # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   # @Description  : Please add descriptioon
### 
   # --scheduler 'CosineAL' \
# For NMNIST A_LIF dvs_input
export CUDA_VISIBLE_DEVICES=1
python ../main.py \
   --model 'BPTT' \
   --structure 'SCN_4' \
   --dataSet 'NMNIST' \
   --dataDir '/opt/data/durian/Benchmark' \
   --resDir '../results' \
   --encoding 'dvs' \
   --nEpoch 100 \
   --batchSize 32 \
   --v_th 1.0 \
   --v_decay -0.8 \
   --T 10 \
   --dt 20 \
   --sur_grad 'linear' \
   --loss_fun 'ce' \
   --optimizer 'adam' \
   --lr 0.0001 \
   --train