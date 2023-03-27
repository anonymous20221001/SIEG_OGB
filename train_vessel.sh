cmd_time=`TZ=UTC-8 date  "+%Y%m%d-%H%M%S"`

# # DGCNN_noNeigFeat
# nohup python3 train.py --device 0 --cmd_time ${cmd_time} --dataset ogbl-vessel --use_feature --use_edge_weight --epochs 20 --train_percent 100 --val_percent 100 --test_percent 100 --model DGCNN_noNeigFeat --runs 10 --batch_size 256 --lr 0.0002 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test >> train_log/run_${cmd_time}.log 2>&1 &

# # DGCNNGraphormer
# nohup python3 train.py --hidden_channels 128 --max_nodes_per_hop 100 --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-vessel --use_feature --use_feature_GT --use_edge_weight --epochs 20 --train_percent 100 --val_percent 100 --test_percent 100 --model DGCNNGraphormer --runs 10 --batch_size 256 --lr 0.0001 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac >> train_log/run_${cmd_time}.log 2>&1 &

# DGCNNGraphormer_noNeigFeat
nohup python3 train.py --hidden_channels 128 --max_nodes_per_hop 100 --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-vessel --use_feature --use_feature_GT --use_edge_weight --epochs 20 --train_percent 100 --val_percent 100 --test_percent 100 --model DGCNNGraphormer_noNeigFeat --runs 10 --batch_size 256 --lr 0.0001 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac >> train_log/run_${cmd_time}.log 2>&1 &

# # SingleGraphormer
# nohup python3 train.py --hidden_channels 128 --max_nodes_per_hop 100 --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-vessel --use_feature --use_feature_GT --use_edge_weight --epochs 10 --train_percent 100 --val_percent 100 --test_percent 100 --model SingleGraphormer --runs 10 --batch_size 256 --lr 0.0002 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac >> train_log/run_${cmd_time}.log 2>&1 &

# # SingleGraphormer_continue_from
# nohup python3 train.py --save_appendix _20230322185425 --continue_from 1 10 --hidden_channels 128 --max_nodes_per_hop 100 --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-vessel --use_feature --use_feature_GT --use_edge_weight --epochs 10 --train_percent 100 --val_percent 100 --test_percent 100 --model SingleGraphormer --runs 10 --batch_size 256 --lr 0.0002 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac >> train_log/run_${cmd_time}.log 2>&1 &
