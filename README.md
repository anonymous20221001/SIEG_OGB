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
| Val results | 82.55%&plusmn;0.40% | 89.48%&plusmn;0.08% | 
| Test results | 82.49%&plusmn;0.41% | 89.57%&plusmn;0.10% |

Usages
------

### ogbl-vessel

```
    python train.py --device 1 --dataset ogbl-vessel --num_hops 1 --use_feature --use_edge_weight --eval_steps 1 --epochs 10 --train_percent 100 --val_percent 100 --test_percent 100 --model DGCNNGraphormer --runs 1 --batch_size 256 --lr 0.0002 --num_workers 24 --sample_type 0 --use_num_spd --use_cnb_jac --use_cnb_aa --sortpool_k 115
```
or
```
    sh train_vessel.sh
```

### ogbl-citation2

```
    python train.py --grpe_cross --device 0 --num_heads 8 --dataset ogbl-citation2 --use_feature --use_feature_GT --use_edge_weight --epochs 15 --train_percent 8 --val_percent 4 --test_percent 1 --final_val_percent 100 --final_test_percent 100 --model SIEG --runs 10 --batch_size 64 --lr 2e-05 --num_workers 24 --sortpool_k 0.6 --dynamic_train --dynamic_val --dynamic_test --use_len_spd --use_num_spd --use_cnb_jac --use_cnb_aa
```
or
```
    sh train_citation2.sh
```

License
-------

SIEG is released under an MIT license. Find out more about it [here](LICENSE).
