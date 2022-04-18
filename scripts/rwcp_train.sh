# coding=utf-8
### 
   # @Author       : Noah
   # @Version      : v1.0.0
   # @Date         : 2020-12-24 10:30:28
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2022-03-29 11:07:12
   # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   # @Description  : Please add descriptioon
### 

# For RWCP A_LIF
export CUDA_VISIBLE_DEVICES=1
python ../main.py \
   --model 'BPTT' \
   --structure 'SNN_2' \
   --dataSet 'RWCP' \
   --dataDir '/opt/data/durian/Benchmark' \
   --resDir '../results' \
   --nEpoch 20 \
   --batchSize 32 \
   --v_th 0.5 \
   --v_decay 0.9 \
   --T 255 \
   --sur_grad 'linear' \
   --loss_fun 'ce' \
   --optimizer 'adam' \
   --lr 0.001 \
   --train

# export CUDA_VISIBLE_DEVICES=1
# python ../main.py \
#    --model 'BPTT' \
#    --structure 'SCN_4v1' \
#    --dataSet 'MNIST' \
#    --dataDir '/opt/data/durian/Benchmark' \
#    --resDir '../results' \
#    --encoding 'fixed' \
#    --nEpoch 240 \
#    --batchSize 32 \
#    --v_th 1.0 \
#    --v_decay -0.8 \
#    --T 8 \
#    --sur_grad 'linear' \
#    --loss_fun 'ce' \
#    --optimizer 'adam' \
#    --scheduler 'CosineAL' \
#    --lr 0.0001 \
#    --train