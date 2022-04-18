# coding=utf-8
### 
   # @Author       : Noah
   # @Version      : v1.0.0
   # @Date         : 2020-12-24 10:30:28
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2021-01-01 11:59:16
   # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   # @Description  : Please add descriptioon
### 

# For RWCP A_LIF
# train_2020-12-28-02_49_28
export CUDA_VISIBLE_DEVICES=1
python ../main.py \
   --model 'BPTT' \
   --structure 'SNN_2v1' \
   --dataSet 'RWCP' \
   --dataDir './Data' \
   --checkpoints '2020-12-28-02_49_28' \
   --resDir '../results' \
   --nEpoch 10 \
   --batchSize 128 \
   --seed 49 \
   --input_size 576 \
   --output_size 10 \
   --v_th 0.5 \
   --v_decay 1.0 \
   --T 255 \
   --step 255 255 \
   --sur_grad 'linear' \
   --loss_fun 'ce' \
   --optimizer 'adam' \
   --lr 0.001