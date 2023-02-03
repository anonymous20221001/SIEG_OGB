cmd_time=`TZ=UTC-8 date  "+%Y%m%d-%H%M%S"`

# DGCNNGraphormer
nohup python3 train.py --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-vessel --use_feature --use_feature_GT --use_edge_weight --epochs 20 --train_percent 100 --val_percent 100 --test_percent 100 --model DGCNNGraphormer --runs 10 --batch_size 256 --lr 0.0001 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac >> train_log/run_${cmd_time}.log 2>&1 &

# # DGCNNGraphormer_noNeigFeat lr0.0001
# nohup python3 train.py --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-vessel --use_feature --use_feature_GT --use_edge_weight --epochs 20 --train_percent 100 --val_percent 100 --test_percent 100 --model DGCNNGraphormer_noNeigFeat --runs 10 --batch_size 256 --lr 0.0001 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac >> train_log/run_${cmd_time}.log 2>&1 &

# # DGL_NGNN
# nohup python3 train.py --ngnn_code --device 0 --cmd_time ${cmd_time} --dataset ogbl-vessel --use_feature --use_edge_weight --epochs 15 --train_percent 100 --val_percent 100 --test_percent 100 --model DGCNN --runs 10 --batch_size 256 --lr 0.0001 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test >> train_log/run_${cmd_time}.log 2>&1 &

# # PYG_NGNN
# nohup python3 train.py --use_ignn --device 1 --cmd_time ${cmd_time} --dataset ogbl-vessel --use_feature --use_edge_weight --epochs 15 --train_percent 100 --val_percent 100 --test_percent 100 --model DGCNN --runs 10 --batch_size 256 --lr 0.0001 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test >> train_log/run_${cmd_time}.log 2>&1 &
