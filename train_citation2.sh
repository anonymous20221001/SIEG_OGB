cmd_time=`TZ=UTC-8 date  "+%Y%m%d-%H%M%S"`

# # NGNNDGCNNGraphormer
# nohup python3 train.py --ngnn_code --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-citation2 --use_feature --use_feature_GT --use_edge_weight --epochs 15 --train_percent 8 --val_percent 4 --test_percent 0.2 --model NGNNDGCNNGraphormer --runs 2 --batch_size 64 --lr 2e-05 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa >> train_log/run_${cmd_time}.log 2>&1 &

# # NGNNDGCNN
# nohup python3 train.py --ngnn_code --device 0 --cmd_time ${cmd_time} --dataset ogbl-citation2 --use_feature --use_edge_weight --epochs 15 --train_percent 8 --val_percent 4 --test_percent 0.2 --model DGCNN --runs 2 --batch_size 64 --lr 2e-05 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test >> train_log/run_${cmd_time}.log 2>&1 &

# # NGNNDGCNN_noNeigFeat
# nohup python3 train.py --ngnn_code --device 0 --cmd_time ${cmd_time} --dataset ogbl-citation2 --use_feature --use_edge_weight --epochs 15 --train_percent 8 --val_percent 4 --test_percent 0.2 --model NGNNDGCNN_noNeigFeat --runs 3 --batch_size 64 --lr 2e-05 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test >> train_log/run_${cmd_time}.log 2>&1 &

# NGNNDGCNNGraphormer_noNeigFeat
nohup python3 train.py --ngnn_code --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-citation2 --use_feature --use_feature_GT --use_edge_weight --epochs 15 --train_percent 8 --val_percent 4 --test_percent 0.2 --model NGNNDGCNNGraphormer_noNeigFeat --runs 10 --batch_size 64 --lr 2e-05 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa >> train_log/run_${cmd_time}.log 2>&1 &

# # SingleGraphormer
# nohup python3 train.py --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-citation2 --use_feature --use_feature_GT --use_edge_weight --epochs 15 --train_percent 8 --val_percent 4 --test_percent 0.2 --model SingleGraphormer --runs 3 --batch_size 64 --lr 2e-05 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa >> train_log/run_${cmd_time}.log 2>&1 &
