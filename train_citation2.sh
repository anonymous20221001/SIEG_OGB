#!/bin/sh

postfix=`TZ=UTC-8 date  "+%Y%m%d-%H%M%S"`

nohup python3 train.py --grpe_cross --device 0 --num_heads 8 --dataset ogbl-citation2 --use_feature --use_feature_GT --use_edge_weight --epochs 15 --train_percent 8 --val_percent 4 --test_percent 1 --final_val_percent 100 --final_test_percent 100 --model SIEG --runs 10 --batch_size 64 --lr 2e-05 --num_workers 24 --sortpool_k 0.6 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa >> citation2_${postfix}.log 2>&1 &
