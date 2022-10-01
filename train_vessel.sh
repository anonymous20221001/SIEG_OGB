#export utils=True
postfix=`TZ=UTC-8 date  "+%Y%m%d-%H%M%S"`

sample_type=0
readout_type=0
#readout_type=2
train_per=100  # 2
per=100  # 1 验证和测试
epochs=10
batch_size=256  # 128
model=DGCNNGraphormer
runs=1  # 10
lr=0.0002
#max_node=100
max_node=-1
num_workers=24
sortpool_k=115

nohup python train.py --device 1 --dataset ogbl-vessel --num_hops 1 --use_feature --use_edge_weight --eval_steps 1 --epochs ${epochs} --train_percent ${train_per} --val_percent ${per} --test_percent ${per} --model ${model} --runs ${runs} --batch_size ${batch_size} --lr ${lr} --num_workers ${num_workers} --sample_type ${sample_type} --use_num_spd --use_cnb_jac --use_cnb_aa --sortpool_k ${sortpool_k} >> vessel_${model}_sampler${sample_type}_sortpoolk${sortpool_k}_b128_lr0.0002_run${runs}_${postfix}_`hostname`.log 2>&1 &
