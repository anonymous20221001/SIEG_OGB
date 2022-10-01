import argparse
import time
import os, sys
import os.path as osp
from shutil import copy
import copy as cp
from tqdm import tqdm
import pdb

import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from utils import *
from models import *
import networkx as nx
from timer_guard import TimerGuard
from collection import Collections

def preprocess(data, **kwargs):
    use_num_spd = kwargs["use_num_spd"] if "use_num_spd" in kwargs else False
    use_cnb_jac = kwargs["use_cnb_jac"] if "use_num_jac" in kwargs else False
    use_cnb_aa = kwargs["use_cnb_aa"] if "use_num_aa" in kwargs else False
    use_degree = kwargs["use_degree"] if "use_degree" in kwargs else False
    directed = kwargs["directed"] if "directed" in kwargs else False

    device = data.x.device
    #_, center_indices = np.unique(data.batch.cpu().numpy(), return_index=True)
    center_indices = np.array([0])
    num_pair = len(center_indices)

    # estimate shortest path using undirected graph
    G = to_networkx(Data(torch.zeros_like(data.node_id), torch.cat([data.edge_index.cpu(), data.edge_index.cpu()[[1, 0], :]], dim=-1)), to_undirected=True)
    pair_len_shortest_path = torch.zeros((num_pair, 2, 2), dtype=torch.long)
    pair_num_shortest_path = torch.zeros((num_pair, 2, 2), dtype=torch.long)
    pair_undir_aa = torch.zeros((num_pair, 2, 2), dtype=torch.float32)
    pair_undir_jac = torch.zeros((num_pair, 2, 2), dtype=torch.float32)
    for i in range(num_pair):
        src_idx = center_indices[i]
        dst_idx = center_indices[i] + 1
        try:
            pair_len_shortest_path[i] = nx.shortest_path_length(G, src_idx, dst_idx)
            if use_num_spd:
                shortest_path_list = [p for p in nx.all_shortest_paths(G, src_idx, dst_idx)]
            if use_cnb_jac:
                preds = nx.jaccard_coefficient(G, [(src_idx, dst_idx)])
                _, _, jac = next(preds)
                pair_undir_jac[i] = jac
            if use_cnb_aa:
                preds = nx.adamic_adar_index(G, [(src_idx, dst_idx)])
                _, _, aa = next(preds)
                pair_undir_aa[i] = aa
        except nx.exception.NetworkXNoPath:
            # No way between these two points
            pair_len_shortest_path[i] = np.iinfo('long').max
            shortest_path_list = []
            pair_num_shortest_path[i] = 0
        pair_len_shortest_path[i].fill_diagonal_(0)
        pair_num_shortest_path[i].fill_diagonal_(0)
        pair_undir_jac[i].fill_diagonal_(0)
        pair_undir_aa[i].fill_diagonal_(0)

    data.pair_len_shortest_path = pair_len_shortest_path.to(device)
    data.pair_num_shortest_path = pair_num_shortest_path.to(device)
    data.pair_undir_jac = pair_undir_jac.to(device)
    data.pair_undir_aa = pair_undir_aa.to(device)

    n_graph = len(center_indices)
    n_node = 2
    data.pair_attn_bias = torch.zeros(n_graph, n_node+1, n_node+1).to(device)
    data.pair_edge_idx = torch.Tensor([[0, 1], [1, 0]]).long().unsqueeze(axis=0).expand(n_graph, -1, -1).to(device)
    x_src = data.x[center_indices]
    x_dst = data.x[center_indices+1]
    data.pair_x = torch.stack((x_src, x_dst), dim=1)

    if use_degree:
        if directed:
            G = to_networkx(Data(torch.zeros_like(data.node_id), edge_index_deleted.cpu()), to_undirected=False)
            data.pair_in_degree = torch.stack((torch.Tensor([v for k, v in dict(G.in_degree(center_indices)).items()]).long(), torch.Tensor([v for k, v in dict(G.in_degree(center_indices+1)).items()]).long()), dim=1).to(device)
            data.pair_out_degree = torch.stack((torch.Tensor([v for k, v in dict(G.out_degree(center_indices)).items()]).long(), torch.Tensor([v for k, v in dict(G.out_degree(center_indices+1)).items()]).long()), dim=1).to(device)
        else:
            data.pair_in_degree = torch.stack((torch.Tensor([v for k, v in dict(G.degree(center_indices)).items()]).long(), torch.Tensor([v for k, v in dict(G.degree(center_indices+1)).items()]).long()), dim=1).to(device)
            data.pair_out_degree = torch.stack((torch.Tensor([v for k, v in dict(G.degree(center_indices)).items()]).long(), torch.Tensor([v for k, v in dict(G.degree(center_indices+1)).items()]).long()), dim=1).to(device)
    else:
        data.pair_in_degree = torch.zeros((n_graph, 2)).to(device)
        data.pair_out_degree = torch.zeros((n_graph, 2)).to(device)

    return data

