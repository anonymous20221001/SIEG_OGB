SIEG
===============================================================================


Code for paper [**SIEG: Structural Information Enhanced Graph representation**](OGB_VESSEL_SIEG.pdf)

This implementation of PairRE for [**Open Graph Benchmak**](https://arxiv.org/abs/2005.00687) datasets (ogbl-wikikg and ogbl-biokg) is based on [**OGB**](https://github.com/snap-stanford/ogb). Thanks for their contributions.

Requirements
------------

Latest tested combination: Python 3.8.5 + PyTorch 1.7.0 + PyTorch\_Geometric 1.7.2 + Networkx 2.5 + OGB 1.3.4.

Install [PyTorch](https://pytorch.org/)

Install [PyTorch\_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Install [Networkx](https://networkx.org/documentation/stable/install.html)

Install [OGB](https://ogb.stanford.edu/docs/home/)

Other required python libraries include: numpy, scipy, tqdm etc.

Result
-----
|              | ogbl-vessel | ogbl-citation2 |
|--------------|---------------------|-----------------------|
| Val results | 83.15%&plusmn;0.44% | 89.78%&plusmn;0.18% | 
| Test results | 83.07%&plusmn;0.44% | 89.87%&plusmn;0.18% |

Usages
------

### ogbl-vessel

```
    python3 train.py --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-vessel --use_feature --use_feature_GT --use_edge_weight --epochs 20 --train_percent 100 --val_percent 100 --test_percent 100 --model DGCNNGraphormer --runs 10 --batch_size 256 --lr 0.0001 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac
```
or
```
    sh train_vessel.sh
```

### ogbl-citation2

```
    python3 train.py --ngnn_code --grpe_cross --device 0 --cmd_time ${cmd_time} --num_heads 8 --dataset ogbl-citation2 --use_feature --use_feature_GT --use_edge_weight --epochs 15 --train_percent 8 --val_percent 4 --test_percent 0.2 --model NGNNDGCNNGraphormer_noNeigFeat --runs 10 --batch_size 64 --lr 2e-05 --num_workers 24 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa
```
or
```
    sh train_citation2.sh
```

License
-------

SIEG is released under an MIT license. Find out more about it [here](LICENSE).
