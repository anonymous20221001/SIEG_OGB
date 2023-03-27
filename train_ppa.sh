cmd_time=`TZ=UTC-8 date  "+%Y%m%d-%H%M%S"`

# # NGNNDGCNN
# nohup python3 train.py --cmd_time ${cmd_time} --eval_hits_K 45 60 75 100 --device 0 --ngnn_code --ngnn_type input --num_ngnn_layers 1 --hidden_channels 96 --dataset ogbl-ppa --use_feature --epochs 4 --train_percent 60 --val_percent 40 --test_percent 1 --model DGCNN --runs 10 --batch_size 128 --lr 0.00015 --num_workers 48 --dynamic_train --dynamic_val --dynamic_test >> train_log/run_${cmd_time}.log 2>&1 &

# # NGNNDGCNN_onlyfinaltest
# nohup python3 train.py --cmd_time ${cmd_time} --save_appendix _20230217113720 --continue_from 1 10 --only_final_test --eval_hits_K 100 --device 0 --ngnn_code --ngnn_type input --num_ngnn_layers 1 --hidden_channels 96 --dataset ogbl-ppa --use_feature --model DGCNN --num_workers 48 --dynamic_train --dynamic_val --dynamic_test >> train_log/run_${cmd_time}.log 2>&1 &

# # NGNNDGCNN_noNeigFeat
# nohup python3 train.py --cmd_time ${cmd_time} --eval_hits_K 45 60 75 100 --device 0 --ngnn_code --ngnn_type input --num_ngnn_layers 1 --hidden_channels 96 --dataset ogbl-ppa --use_feature --epochs 4 --train_percent 60 --val_percent 40 --test_percent 1 --model NGNNDGCNN_noNeigFeat --runs 10 --batch_size 128 --lr 0.00015 --num_workers 48 --dynamic_train --dynamic_val --dynamic_test >> train_log/run_${cmd_time}.log 2>&1 &

# # NGNNDGCNNGraphormer
# nohup python3 train.py --cmd_time ${cmd_time} --eval_hits_K 45 60 75 100 --device 0 --ngnn_code --grpe_cross --num_heads 8 --ngnn_type input --num_ngnn_layers 1 --hidden_channels 96 --dataset ogbl-ppa --use_feature --use_feature_GT --epochs 4 --train_percent 60 --val_percent 40 --test_percent 1 --model NGNNDGCNNGraphormer --runs 10 --batch_size 128 --lr 0.00015 --num_workers 48 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa --use_cnb_ra >> train_log/run_${cmd_time}.log 2>&1 &

# # NGNNDGCNNGraphormer_onlyfinaltest
# nohup python3 train.py --cmd_time ${cmd_time} --save_appendix _20230217154015 --continue_from 1 1001 --only_final_test --eval_hits_K 100 --device 0 --ngnn_code --grpe_cross --num_heads 8 --ngnn_type input --num_ngnn_layers 1 --hidden_channels 96 --dataset ogbl-ppa --use_feature --use_feature_GT --model NGNNDGCNNGraphormer --num_workers 48 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa --use_cnb_ra >> train_log/run_${cmd_time}.log 2>&1 &

# # NGNNDGCNNGraphormer_from_NGNNDGCNN
# nohup python3 train.py --cmd_time ${cmd_time} --save_appendix _20230217113720 --part_continue_from 1 10 --eval_hits_K 6 8 10 100 --device 0 --ngnn_code --grpe_cross --num_heads 8 --ngnn_type input --num_ngnn_layers 1 --hidden_channels 96 --dataset ogbl-ppa --use_feature --use_feature_GT --epochs 4 --train_percent 8 --val_percent 8 --test_percent 1 --model NGNNDGCNNGraphormer --runs 10 --batch_size 128 --lr 0.00015 --num_workers 48 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa --use_cnb_ra >> train_log/run_${cmd_time}.log 2>&1 &

# NGNNDGCNNGraphormer_noNeigFeat
nohup python3 train.py --cmd_time ${cmd_time} --eval_hits_K 45 60 75 100 --device 0 --ngnn_code --grpe_cross --num_heads 8 --ngnn_type input --num_ngnn_layers 1 --hidden_channels 96 --dataset ogbl-ppa --use_feature --use_feature_GT --epochs 4 --train_percent 60 --val_percent 40 --test_percent 1 --model NGNNDGCNNGraphormer_noNeigFeat --runs 10 --batch_size 128 --lr 0.00015 --num_workers 48 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa --use_cnb_ra >> train_log/run_${cmd_time}.log 2>&1 &

# # NGNNDGCNNGraphormer_noNeigFeat epochs16 train_per16 val_per16 bs256 lr0.0002 ns64 eval_hits_K 12 16 20 24 100 hidden_channels 144
# nohup python3 train.py --cmd_time ${cmd_time} --eval_hits_K 12 16 20 24 100 --device 0 --ngnn_code --grpe_cross --num_heads 8 --ngnn_type input --num_ngnn_layers 1 --hidden_channels 144 --dataset ogbl-ppa --use_feature --use_feature_GT --epochs 16 --train_percent 16 --val_percent 16 --test_percent 1 --model NGNNDGCNNGraphormer_noNeigFeat --runs 10 --batch_size 256 --lr 0.0002 --num_workers 64 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa --use_cnb_ra >> train_log/run_${cmd_time}.log 2>&1 &

# # NGNNDGCNNGraphormer_noNeigFeat_from_NGNNDGCNN_noNeigFeat
# nohup python3 train.py --cmd_time ${cmd_time} --save_appendix _20230217113720 --part_continue_from 1 10 --eval_hits_K 6 8 10 100 --device 0 --ngnn_code --grpe_cross --num_heads 8 --ngnn_type input --num_ngnn_layers 1 --hidden_channels 96 --dataset ogbl-ppa --use_feature --use_feature_GT --epochs 4 --train_percent 8 --val_percent 8 --test_percent 1 --model NGNNDGCNNGraphormer_noNeigFeat --runs 10 --batch_size 128 --lr 0.00015 --num_workers 48 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa --use_cnb_ra >> train_log/run_${cmd_time}.log 2>&1 &

# # SingleGraphormer
# nohup python3 train.py --cmd_time ${cmd_time} --eval_hits_K 12 16 20 24 100 --device 0 --grpe_cross --num_heads 8 --dataset ogbl-ppa --use_feature --use_feature_GT --epochs 4 --train_percent 16 --val_percent 16 --test_percent 1 --model SingleGraphormer --runs 10 --batch_size 128 --lr 0.0004 --num_workers 48 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_ra >> train_log/run_${cmd_time}.log 2>&1 &

# # SingleGraphormer_onlyfinaltest
# nohup python3 train.py --cmd_time ${cmd_time} --save_appendix _20230321195654 --continue_from 1 3 --only_final_test --eval_hits_K 100 --device 0 --grpe_cross --num_heads 8 --dataset ogbl-ppa --use_feature --use_feature_GT --model SingleGraphormer --runs 10 --batch_size 128 --lr 0.0004 --num_workers 48 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_ra >> train_log/run_${cmd_time}.log 2>&1 &


# GCN
#nohup python3 train.py --cmd_time ${cmd_time} --device 0 --num_layers 2 --hidden_channels 16 --dataset ogbl-ppa --use_feature --epochs 100 --train_percent 2 --val_percent 2 --test_percent 1 --final_val_percent 2 --final_test_percent 1 --model GCN --runs 100 --batch_size 128 --lr 0.00015 --num_workers 16 --dynamic_train --dynamic_val --dynamic_test >> train_log/run_${cmd_time}.log 2>&1 &
