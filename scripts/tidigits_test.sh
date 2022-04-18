# coding=utf-8
### 
   # @Author       : Noah
   # @Version      : v1.0.0
   # @Date         : 2020-12-24 10:30:28
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2020-12-31 09:43:59
   # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   # @Description  : Please add descriptioon
### 

# For TIDIGITS A_LIF
export CUDA_VISIBLE_DEVICES=1
python ../main.py \
   --model 'BPTT' \
   --structure 'SNN_2v1' \
   --dataSet 'TIDIGITS' \
   --dataDir './Data' \
   --resDir '../results' \
   --checkpoints '2020-12-27-08_08_25' \
   --batchSize 128 \
   --seed 49 \
   --input_size 576 \
   --output_size 11 \
   --v_th 0.5 \
   --v_decay 1.0 \
   --T 80 \
   --step 80 80 \
   --sur_grad 'linear' \
   --loss_fun 'ce' \
   --optimizer 'adam' \
   --lr 0.001