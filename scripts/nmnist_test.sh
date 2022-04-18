# coding=utf-8
### 
   # @Author       : Noah
   # @Version      : v1.0.0
   # @Date         : 2020-12-24 10:30:28
 # @LastEditors  : Please set LastEditors
 # @LastEditTime : 2021-01-01 11:58:20
   # @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   # @Description  : Please add descriptioon
### 

# For NMNIST A_LIF dvs_input
export CUDA_VISIBLE_DEVICES=1
python ../main.py \
   --model 'BPTT' \
   --structure 'SCN_3v1' \
   --dataSet 'NMNIST' \
   --dataDir './Data' \
   --resDir '../results' \
   --checkpoints '2020-12-27-08_29_48' \
   --encoding 'dvs' \
   --nEpoch 100 \
   --batchSize 100 \
   --seed 49 \
   --img_size 34 34 1 \
   --input_size 1156 \
   --output_size 10 \
   --v_th 0.5 \
   --v_decay 1.0 \
   --T 12 \
   --step 12 12 \
   --sur_grad 'linear' \
   --loss_fun 'ce' \
   --optimizer 'adam' \
   --lr 0.001