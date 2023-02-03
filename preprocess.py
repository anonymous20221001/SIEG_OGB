# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import os, sys
import os.path as osp
from shutil import copy
import copy as cp
from tqdm import tqdm
import pdb

import numpy as np
from functools import reduce
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


@TimerGuard('process_paths3', 'utils')
def process_paths3(paths, edge_index, edge_types, edge_degrees, num_edge_type, directed):
    num_paths = len(paths)
    paths_edge_types = [[] for i in range(num_paths)]
    paths_type = [0 for i in range(num_paths)]
    paths_weight = [0 for i in range(num_paths)]
    paths_degrees = [[] for i in range(num_paths)]
    paths_depth = [0 for i in range(num_paths)]
    paths_gravity = [0 for i in range(num_paths)]
    for idx_path in range(num_paths):
        path = paths[idx_path]
        if path is None:
            pdb.set_trace()
        path_depth = len(path)-1
        path_edge_types = [[] for i in range(path_depth)]
        path_degrees = [[] for i in range(path_depth)]
        path_weight = 1.0
        for idx_edge in range(path_depth):
            node_id = path[idx_edge]
            if directed:
                mask = torch.where((edge_index == torch.Tensor([[path[idx_edge]], [path[idx_edge+1]]]).long()).all(axis=0))[0]
                if len(mask.tolist()) > 2:
                    pdb.set_trace()
                mask = mask[0]
                path_edge_types[idx_edge] = edge_types[mask].item()
                path_degrees[idx_edge] = edge_degrees[mask].item()
            path_weight /= path_degrees[idx_edge]
        paths_edge_types[idx_path] = path_edge_types
        path_type = pow(num_edge_type*2, path_depth) - 2
        for idx_edge, edge_type in enumerate(path_edge_types):
            path_type += edge_type * pow(num_edge_type*2, path_depth-1-idx_edge)
        paths_type[idx_path] = path_type
        paths_weight[idx_path] = path_weight
        paths_degrees[idx_path] = path_degrees

        paths_depth[idx_path] = path_depth
        paths_gravity[idx_path] = pow(path_depth, -2)

    return paths_edge_types, paths_type, paths_weight, paths_degrees, paths_gravity

@TimerGuard('process_paths2', 'utils')
def process_paths2(paths, node_ids, G_undirected, degree, num_edge_type, directed):
    num_paths = len(paths)
    paths_edge_types = [[] for i in range(num_paths)]
    paths_type = [0 for i in range(num_paths)]
    paths_weight = [0 for i in range(num_paths)]
    paths_degrees = [[] for i in range(num_paths)]
    paths_depth = [0 for i in range(num_paths)]
    paths_gravity = [0 for i in range(num_paths)]
    for idx_path in range(num_paths):
        path = paths[idx_path]
        if path is None:
            pdb.set_trace()
        path_depth = len(path)-1
        path_edge_types = [[] for i in range(path_depth)]
        path_degrees = [[] for i in range(path_depth)]
        path_weight = 1.0
        for idx_edge in range(path_depth):
            node_id_global = node_ids[path[idx_edge]]

            node_id = path[idx_edge]
            path_edge = G_undirected.get_edge_data(path[idx_edge], path[idx_edge+1])
            path_edge_types[idx_edge], path_degrees[idx_edge] = path_edge['edge_attrs']
            #path_edge_types[idx_edge] = path_edge['edge_type']
            #path_degrees[idx_edge] = path_edge['edge_degree']
            if path_degrees[idx_edge] == 0:
                pdb.set_trace()
                print(degree[:, node_id_global])
            #print(f'{idx_path} {idx_edge}')
            #print(degree[:, node_id_global], path_edge['edge_degree'])
            path_weight /= path_degrees[idx_edge]
        paths_edge_types[idx_path] = path_edge_types
        path_type = pow(num_edge_type*2, path_depth) - 2
        for idx_edge, edge_type in enumerate(path_edge_types):
            path_type += edge_type * pow(num_edge_type*2, path_depth-1-idx_edge)
        paths_type[idx_path] = path_type
        paths_weight[idx_path] = path_weight
        paths_degrees[idx_path] = path_degrees

        paths_depth[idx_path] = path_depth
        paths_gravity[idx_path] = pow(path_depth, -2)

    return paths_edge_types, paths_type, paths_weight, paths_degrees, paths_gravity

@TimerGuard('process_paths', 'utils')
def process_paths(paths, G, G_inv, degree, num_edge_type, directed):
    num_paths = len(paths)
    paths_edge_types = []
    paths_type = []
    paths_weight = []
    paths_degrees = []
    paths_depth = []
    paths_gravity = []
    for idx_path in range(num_paths):
        path = paths[idx_path]
        if path is None:
            pdb.set_trace()
        path_edge_types = []
        path_degrees = []
        path_depth = len(path)-1
        for idx_edge in range(path_depth):
            if directed:
                if G.has_edge(path[idx_edge], path[idx_edge+1]):
                    path_edge_types.append(G.get_edge_data(path[idx_edge], path[idx_edge+1])['edge_type'])
                    node_id = path[idx_edge]
                    path_degrees.append(degree[2, node_id].item())
                else:
                    path_edge_types.append(G_inv.get_edge_data(path[idx_edge], path[idx_edge+1])['edge_type'])
                    node_id = path[idx_edge]
                    path_degrees.append(degree[1, node_id].item())
            else:
                path_edge_types.append(G_undirected.get_edge_data(path[idx_edge], path[idx_edge+1])['edge_type'])
                node_id = path[idx_edge]
                path_degrees.append(degree[0, node_id].item())
        paths_edge_types.append(path_edge_types)
        path_type = pow(num_edge_type*2, path_depth) - 2
        path_weight = 1.0
        for idx_edge, edge_type in enumerate(path_edge_types):
            path_type += edge_type * pow(num_edge_type*2, path_depth-1-idx_edge)
            path_weight /= path_degrees[idx_edge]
        paths_type.append(path_type)
        paths_weight.append(path_weight)
        paths_degrees.append(path_degrees)

        paths_depth.append(path_depth)
        paths_gravity.append(pow(path_depth, -2))

    return paths_edge_types, paths_type, paths_weight, paths_degrees, paths_gravity

@TimerGuard('preprocess', 'utils')
def preprocess(data, degree, **kwargs):
    #print(data)
    #print(kwargs)
    # import pdb; pdb.set_trace()
    use_len_spd = kwargs["use_len_spd"] if "use_len_spd" in kwargs else False
    use_num_spd = kwargs["use_num_spd"] if "use_num_spd" in kwargs else False
    use_cnb_jac = kwargs["use_cnb_jac"] if "use_cnb_jac" in kwargs else False
    use_cnb_aa = kwargs["use_cnb_aa"] if "use_cnb_aa" in kwargs else False
    use_cnb_ra = kwargs["use_cnb_ra"] if "use_cnb_ra" in kwargs else False
    use_degree = kwargs["use_degree"] if "use_degree" in kwargs else False
    gravity_type = kwargs["gravity_type"] if "gravity_type" in kwargs else 0
    directed = kwargs["directed"] if "directed" in kwargs else False
    #print(f'use_len_spd={use_len_spd}')
    #print(f'use_num_spd={use_num_spd}')
    #print(f'use_cnb_jac={use_cnb_jac}')
    #print(f'use_cnb_aa={use_cnb_aa}')
    #print(f'use_cnb_ra={use_cnb_ra}')
    #print(f'use_degree={use_degree}')
    #print(f'gravity_type={gravity_type}')
    #print(f'directed={directed}')

    device = data.edge_index.device
    center_indices = np.array([0])
    num_pair = len(center_indices)

    # estimate shortest path using undirected graph
    if gravity_type == 0:
        G_undirected = to_networkx(Data(torch.zeros_like(data.node_id), torch.cat([data.edge_index.cpu(), data.edge_index.cpu()[[1, 0], :]], dim=-1)), to_undirected=True)
    elif gravity_type > 0:
        num_edges = data.edge_index.size()[1]
        num_edge_type = 1
        max_len_rule = 3
        num_rules = pow(num_edge_type*2, max_len_rule+1) - 2
        edge_type = torch.tensor([0 for i in range(num_edges)], dtype=torch.long)

        G_undirected = to_networkx(Data(torch.zeros_like(data.node_id), torch.cat([data.edge_index.cpu(), data.edge_index.cpu()[[1, 0], :]], dim=-1)), to_undirected=True)
        if directed:
            G = to_networkx(Data(torch.zeros_like(data.node_id), data.edge_index.cpu(), edge_type=edge_type.numpy()), edge_attrs=['edge_type'], to_undirected=False)
            G_inv = to_networkx(Data(torch.zeros_like(data.node_id), data.edge_index.cpu()[[1, 0], :], edge_type=edge_type.numpy()+num_edge_type), edge_attrs=['edge_type'], to_undirected=False)

    pair_len_shortest_path = torch.zeros((num_pair, 2, 2), dtype=torch.long)
    pair_num_shortest_path = torch.zeros((num_pair, 2, 2), dtype=torch.long)
    pair_undir_jac = torch.zeros((num_pair, 2, 2), dtype=torch.float32)
    pair_undir_aa = torch.zeros((num_pair, 2, 2), dtype=torch.float32)
    pair_undir_ra = torch.zeros((num_pair, 2, 2), dtype=torch.float32)
    if gravity_type > 0:
        pair_paths_weight = torch.zeros((num_pair, 2, 2, num_rules), dtype=torch.float)
        pair_paths_log_weight = torch.zeros((num_pair, 2, 2, num_rules), dtype=torch.float)
        pair_rules_num = torch.zeros((num_pair, 2, 2, num_rules), dtype=torch.long)
    for i in range(num_pair):
        src_idx = center_indices[i]
        dst_idx = center_indices[i] + 1
        if gravity_type > 0:
            paths = list(nx.all_simple_paths(G_undirected, source=src_idx, target=dst_idx, cutoff=max_len_rule))
            paths_inv = cp.deepcopy(paths)
            for path in paths_inv:
                path.reverse()
            num_paths = len(paths)
            #print(f'num paths {num_paths}')
            node_degree = degree[:, data.node_id]
            paths_edge_types, paths_type, paths_weight, paths_degrees, paths_gravity = process_paths(paths, G, G_inv, node_degree, num_edge_type, directed)
            paths_edge_types_inv, paths_type_inv, paths_weight_inv, paths_degrees_inv, paths_gravity_inv = process_paths(paths_inv, G, G_inv, node_degree, num_edge_type, directed)
            #print(paths_edge_types, paths_type, paths_weight, paths_degrees, paths_gravity)
            #print(paths_edge_types_inv, paths_type_inv, paths_weight_inv, paths_degrees_inv, paths_gravity)
            for idx_path in range(num_paths):
                path_weights = 1. / torch.Tensor([degree for degree in paths_degrees[idx_path]])
                path_weights_inv = 1. / torch.Tensor([degree for degree in paths_degrees_inv[idx_path]])
                path_log_weights = 1. / torch.Tensor([math.log(degree, 1.8) for degree in paths_degrees[idx_path]])
                path_log_weights_inv = 1. / torch.Tensor([math.log(degree, 1.8) for degree in paths_degrees_inv[idx_path]])
                path_log_weights[path_log_weights.isinf()] = 1.0
                path_log_weights_inv[path_log_weights_inv.isinf()] = 1.0
                pair_paths_weight[i][1][0][paths_type[idx_path]] += torch.prod(path_weights)
                pair_paths_weight[i][0][1][paths_type_inv[idx_path]] += torch.prod(path_weights_inv)
                pair_paths_log_weight[i][1][0][paths_type[idx_path]] += torch.prod(path_log_weights)
                pair_paths_log_weight[i][0][1][paths_type_inv[idx_path]] += torch.prod(path_log_weights_inv)
                pair_rules_num[i][1][0][paths_type[idx_path]] += 1
                pair_rules_num[i][0][1][paths_type_inv[idx_path]] += 1
        try:
            if use_len_spd:
                pair_len_shortest_path[i] = nx.shortest_path_length(G_undirected, src_idx, dst_idx)
            if use_num_spd:
                shortest_path_list = [p for p in nx.all_shortest_paths(G_undirected, src_idx, dst_idx)]
                pair_num_shortest_path[i] = len(shortest_path_list)
            if use_cnb_jac:
                preds = nx.jaccard_coefficient(G_undirected, [(src_idx, dst_idx)])
                _, _, jac = next(preds)
                pair_undir_jac[i] = jac
            if use_cnb_aa:
                preds = nx.adamic_adar_index(G_undirected, [(src_idx, dst_idx)])
                _, _, aa = next(preds)
                pair_undir_aa[i] = aa
            if use_cnb_ra:
                preds = nx.resource_allocation_index(G_undirected, [(src_idx, dst_idx)])
                _, _, ra = next(preds)
                pair_undir_ra[i] = ra
        except nx.exception.NetworkXNoPath:
            # No way between these two points
            pair_len_shortest_path[i] = np.iinfo('long').max
            shortest_path_list = []
            pair_num_shortest_path[i] = 0
        pair_len_shortest_path[i].fill_diagonal_(0)
        pair_num_shortest_path[i].fill_diagonal_(0)
        pair_undir_jac[i].fill_diagonal_(0)
        pair_undir_aa[i].fill_diagonal_(0)
        pair_undir_ra[i].fill_diagonal_(0)

    if use_len_spd:
        data.pair_len_shortest_path = pair_len_shortest_path.to(device)
    if use_num_spd:
        data.pair_num_shortest_path = pair_num_shortest_path.to(device)
    if use_cnb_jac:
        data.pair_undir_jac = pair_undir_jac.to(device)
    if use_cnb_aa:
        data.pair_undir_aa = pair_undir_aa.to(device)
    if use_cnb_ra:
        data.pair_undir_ra = pair_undir_ra.to(device)
    if gravity_type > 0:
        data.pair_paths_weight = pair_paths_weight.to(device)
        data.pair_paths_log_weight = pair_paths_log_weight.to(device)
        data.pair_rules_num = pair_rules_num.to(device)

    n_graph = len(center_indices)
    n_node = 2
    data.pair_attn_bias = torch.zeros(n_graph, n_node+1, n_node+1).to(device)
    data.pair_edge_idx = torch.Tensor([[0, 1], [1, 0]]).long().unsqueeze(axis=0).expand(n_graph, -1, -1).to(device)
    if data.x is not None:
        x_src = data.x[center_indices]
        x_dst = data.x[center_indices+1]
        data.pair_x = torch.stack((x_src, x_dst), dim=1)
    z_src = data.z[center_indices]
    z_dst = data.z[center_indices+1]
    data.pair_z = torch.stack((z_src, z_dst), dim=1)

    if use_degree:
        data.pair_undir_degree = torch.stack((degree[0, data.node_id[center_indices]], degree[0, data.node_id[center_indices+1]]), dim=1).to(device)
        if directed:
        #    data.pair_in_degree = torch.stack((torch.Tensor([v for k, v in dict(G.in_degree(center_indices)).items()]).long(), torch.Tensor([v for k, v in dict(G.in_degree(center_indices+1)).items()]).long()), dim=1).to(device)
        #    data.pair_out_degree = torch.stack((torch.Tensor([v for k, v in dict(G_inv.in_degree(center_indices)).items()]).long(), torch.Tensor([v for k, v in dict(G_inv.in_degree(center_indices+1)).items()]).long()), dim=1).to(device)
        #else:
        #    data.pair_in_degree = torch.stack((torch.Tensor([v for k, v in dict(G_undirected.degree(center_indices)).items()]).long(), torch.Tensor([v for k, v in dict(G_undirected.degree(center_indices+1)).items()]).long()), dim=1).to(device)
        #    data.pair_out_degree = data.pair_in_degree
            data.pair_in_degree = torch.stack((degree[1, data.node_id[center_indices]], degree[1, data.node_id[center_indices+1]]), dim=1).to(device)
            data.pair_out_degree = torch.stack((degree[2, data.node_id[center_indices]], degree[2, data.node_id[center_indices+1]]), dim=1).to(device)

    #for key in data.keys:
    #    if key.find('pair') != -1:
    #        print(key, data[key])
    return data

@TimerGuard('preprocess_full', 'utils')
def preprocess_full(data, degree, **kwargs):
    #print(data)
    #print(kwargs)
    # import pdb; pdb.set_trace()
    use_len_spd = kwargs["use_len_spd"] if "use_len_spd" in kwargs else False
    use_num_spd = kwargs["use_num_spd"] if "use_num_spd" in kwargs else False
    use_cnb_jac = kwargs["use_cnb_jac"] if "use_cnb_jac" in kwargs else False
    use_cnb_aa = kwargs["use_cnb_aa"] if "use_cnb_aa" in kwargs else False
    use_cnb_ra = kwargs["use_cnb_ra"] if "use_cnb_ra" in kwargs else False
    use_degree = kwargs["use_degree"] if "use_degree" in kwargs else False
    gravity_type = kwargs["gravity_type"] if "gravity_type" in kwargs else 0
    directed = kwargs["directed"] if "directed" in kwargs else False
    #print(f'use_len_spd={use_len_spd}')
    #print(f'use_num_spd={use_num_spd}')
    #print(f'use_cnb_jac={use_cnb_jac}')
    #print(f'use_cnb_aa={use_cnb_aa}')
    #print(f'use_cnb_ra={use_cnb_ra}')
    #print(f'use_degree={use_degree}')
    #print(f'gravity_type={gravity_type}')
    #print(f'directed={directed}')

    device = data.edge_index.device
    center_indices = np.array([0])
    num_pair = len(center_indices)
    assert num_pair == 1
    num_node = data.node_id.shape[0]

    # estimate shortest path using undirected graph
    if gravity_type == 0:
        G_undirected = to_networkx(Data(torch.zeros_like(data.node_id), torch.cat([data.edge_index.cpu(), data.edge_index.cpu()[[1, 0], :]], dim=-1)), to_undirected=True)
    elif gravity_type > 0:
        num_edges = data.edge_index.size()[1]
        num_edge_type = 1
        max_len_rule = 3
        num_rules = pow(num_edge_type*2, max_len_rule+1) - 2
        edge_type = torch.tensor([0 for i in range(num_edges)], dtype=torch.long)

        G_undirected = to_networkx(Data(torch.zeros_like(data.node_id), torch.cat([data.edge_index.cpu(), data.edge_index.cpu()[[1, 0], :]], dim=-1)), to_undirected=True)
        if directed:
            G = to_networkx(Data(torch.zeros_like(data.node_id), data.edge_index.cpu(), edge_type=edge_type.numpy()), edge_attrs=['edge_type'], to_undirected=False)
            G_inv = to_networkx(Data(torch.zeros_like(data.node_id), data.edge_index.cpu()[[1, 0], :], edge_type=edge_type.numpy()+num_edge_type), edge_attrs=['edge_type'], to_undirected=False)

    pair_len_shortest_path = torch.zeros((num_pair, num_node, num_node), dtype=torch.long)
    pair_num_shortest_path = torch.zeros((num_pair, num_node, num_node), dtype=torch.long)
    pair_undir_jac = torch.zeros((num_pair, num_node, num_node), dtype=torch.float32)
    pair_undir_aa = torch.zeros((num_pair, num_node, num_node), dtype=torch.float32)
    pair_undir_ra = torch.zeros((num_pair, num_node, num_node), dtype=torch.float32)

    if gravity_type > 0:
        pair_paths_weight = torch.zeros((num_pair, num_node, num_node, num_rules), dtype=torch.float)
        pair_paths_log_weight = torch.zeros((num_pair, num_node, num_node, num_rules), dtype=torch.float)
        pair_rules_num = torch.zeros((num_pair, num_node, num_node, num_rules), dtype=torch.long)
    idx = 0
    for i in range(1, num_node):
        for j in range(i+1, num_node):
            s_idx = i
            o_idx = j
            if gravity_type > 0:
                paths = list(nx.all_simple_paths(G_undirected, source=s_idx, target=o_idx, cutoff=max_len_rule))
                paths_inv = cp.deepcopy(paths)
                for path in paths_inv:
                    path.reverse()
                num_paths = len(paths)
                #print(f'num paths {num_paths}')
                node_degree = degree[:, data.node_id]
                paths_edge_types, paths_type, paths_weight, paths_degrees, paths_gravity = process_paths(paths, G, G_inv, node_degree, num_edge_type, directed)
                paths_edge_types_inv, paths_type_inv, paths_weight_inv, paths_degrees_inv, paths_gravity_inv = process_paths(paths_inv, G, G_inv, node_degree, num_edge_type, directed)
                #print(paths_edge_types, paths_type, paths_weight, paths_degrees, paths_gravity)
                #print(paths_edge_types_inv, paths_type_inv, paths_weight_inv, paths_degrees_inv, paths_gravity)
                for idx_path in range(num_paths):
                    path_weights = 1. / torch.Tensor([degree for degree in paths_degrees[idx_path]])
                    path_weights_inv = 1. / torch.Tensor([degree for degree in paths_degrees_inv[idx_path]])
                    path_log_weights = 1. / torch.Tensor([math.log(degree, 1.8) for degree in paths_degrees[idx_path]])
                    path_log_weights_inv = 1. / torch.Tensor([math.log(degree, 1.8) for degree in paths_degrees_inv[idx_path]])
                    path_log_weights[path_log_weights.isinf()] = 1.0
                    path_log_weights_inv[path_log_weights_inv.isinf()] = 1.0
                    pair_paths_weight[idx][s_idx][o_idx][paths_type[idx_path]] += torch.prod(path_weights)
                    pair_paths_weight[idx][o_idx][s_idx][paths_type_inv[idx_path]] += torch.prod(path_weights_inv)
                    pair_paths_log_weight[idx][s_idx][o_idx][paths_type[idx_path]] += torch.prod(path_log_weights)
                    pair_paths_log_weight[idx][o_idx][s_idx][paths_type_inv[idx_path]] += torch.prod(path_log_weights_inv)
                    pair_rules_num[idx][s_idx][o_idx][paths_type[idx_path]] += 1
                    pair_rules_num[idx][o_idx][s_idx][paths_type_inv[idx_path]] += 1
            try:
                if use_len_spd:
                    pair_len_shortest_path[idx][s_idx][o_idx] = nx.shortest_path_length(G_undirected, s_idx, o_idx)
                    pair_len_shortest_path[idx][o_idx][s_idx] = nx.shortest_path_length(G_undirected, s_idx, o_idx)
                if use_num_spd:
                    shortest_path_list = [p for p in nx.all_shortest_paths(G_undirected, s_idx, o_idx)]
                    pair_num_shortest_path[idx][s_idx][o_idx] = len(shortest_path_list)
                    pair_num_shortest_path[idx][o_idx][s_idx] = len(shortest_path_list)
                if use_cnb_jac:
                    preds = nx.jaccard_coefficient(G_undirected, [(s_idx, o_idx)])
                    _, _, jac = next(preds)
                    pair_undir_jac[idx][s_idx][o_idx] = jac
                    pair_undir_jac[idx][o_idx][s_idx] = jac
                if use_cnb_aa:
                    preds = nx.adamic_adar_index(G_undirected, [(s_idx, o_idx)])
                    _, _, aa = next(preds)
                    pair_undir_aa[idx][s_idx][o_idx] = aa
                    pair_undir_aa[idx][o_idx][s_idx] = aa
                if use_cnb_ra:
                    preds = nx.resource_allocation_index(G_undirected, [(s_idx, o_idx)])
                    _, _, ra = next(preds)
                    pair_undir_ra[idx][s_idx][o_idx] = ra
                    pair_undir_ra[idx][o_idx][s_idx] = ra
            except nx.exception.NetworkXNoPath:
                # No way between these two points
                pair_len_shortest_path[idx][s_idx][o_idx] = np.iinfo('long').max
                pair_len_shortest_path[idx][o_idx][s_idx] = np.iinfo('long').max
                shortest_path_list = []
                pair_num_shortest_path[idx][s_idx][o_idx] = 0
                pair_num_shortest_path[idx][o_idx][s_idx] = 0
            pair_len_shortest_path[idx].fill_diagonal_(0)
            pair_num_shortest_path[idx].fill_diagonal_(0)
            pair_undir_jac[idx].fill_diagonal_(0)
            pair_undir_aa[idx].fill_diagonal_(0)
            pair_undir_ra[idx].fill_diagonal_(0)

    if use_len_spd:
        data.pair_len_shortest_path = pair_len_shortest_path.to(device)
    if use_num_spd:
        data.pair_num_shortest_path = pair_num_shortest_path.to(device)
    if use_cnb_jac:
        data.pair_undir_jac = pair_undir_jac.to(device)
    if use_cnb_aa:
        data.pair_undir_aa = pair_undir_aa.to(device)
    if use_cnb_ra:
        data.pair_undir_ra = pair_undir_ra.to(device)
    if gravity_type > 0:
        data.pair_paths_weight = pair_paths_weight.to(device)
        data.pair_paths_log_weight = pair_paths_log_weight.to(device)
        data.pair_rules_num = pair_rules_num.to(device)

    if use_degree:
        data.pair_undir_degree = degree[0, :].to(device)
        if directed:
            data.pair_in_degree = degree[1, :].to(device)
            data.pair_out_degree = degree[2, :].to(device)

    n_graph = len(center_indices)
    data.pair_x = data.x.unsqueeze(axis=0)
    data.pair_z = data.z.unsqueeze(axis=0)
    data.pair_edge_idx = data.edge_index.unsqueeze(axis=0).expand(n_graph, -1, -1).to(device)
    data.pair_attn_bias = torch.zeros(n_graph, num_node+1, num_node+1).to(device)

    return data

@TimerGuard('main', 'utils')
def main(data, degree):
    use_len_spd = True
    use_num_spd = True
    use_cnb_jac = True
    use_cnb_aa = True
    use_cnb_ra = True
    use_degree = True
    gravity_type = 2
    directed = True

    device = data.edge_index.device
    center_indices = np.array([0])
    num_pair = len(center_indices)
    i = 0
    src_idx = center_indices[i]
    dst_idx = center_indices[i] + 1

    data.degree = degree[:, data.node_id]

    # remove symmetric edges
    if directed:
        edge_index = data.edge_index.t().numpy().copy()  # (30387995, 2)
        edge_index_inv = edge_index[:, [1, 0]].copy()
        dtype = [('', edge_index.dtype)] * edge_index.shape[0 if edge_index.flags['F_CONTIGUOUS'] else -1]
        edge_index_repeat, x_ind, y_ind = np.intersect1d(edge_index.view(dtype), edge_index_inv.view(dtype), return_indices=True)
        indices = np.array([i for i in range(edge_index.shape[0])])
        indices = np.delete(indices, x_ind)

    num_edges = data.edge_index.size()[1]
    num_edge_type = 1
    max_len_rule = 3
    num_rules = pow(num_edge_type*2, max_len_rule+1) - 2

    edge_type = torch.tensor([0 for i in range(num_edges)], dtype=torch.long)
    with TimerGuard('to_networkx', 'utils'):
        # cost 1500+ms, need to be optimized
        G_undirected = to_networkx(Data(torch.zeros_like(data.node_id), torch.cat([data.edge_index.cpu(), data.edge_index.cpu()[[1, 0], :]], dim=-1)), to_undirected=True)
    with TimerGuard('to_networkx', 'utils'):
        if directed:
            G = to_networkx(Data(torch.zeros_like(data.node_id), data.edge_index.cpu(), edge_type=edge_type.numpy()), edge_attrs=['edge_type'], to_undirected=False)
            G_inv = to_networkx(Data(torch.zeros_like(data.node_id), data.edge_index.cpu()[[1, 0], :], edge_type=edge_type.numpy()+num_edge_type), edge_attrs=['edge_type'], to_undirected=False)
    with TimerGuard('all_simple_paths', 'utils'):
        paths = list(nx.all_simple_paths(G_undirected, source=src_idx, target=dst_idx, cutoff=max_len_rule))
        paths = sorted(paths)
        paths_inv = cp.deepcopy(paths)
        for path in paths_inv:
            path.reverse()
    num_paths = len(paths)

    #print(f'num paths {num_paths}')
    paths_edge_types, paths_type, paths_weight, paths_degrees, paths_gravity = process_paths(paths, G, G_inv, data.degree, num_edge_type, directed)
    paths_edge_types_inv, paths_type_inv, paths_weight_inv, paths_degrees_inv, paths_gravity_inv = process_paths(paths_inv, G, G_inv, data.degree, num_edge_type, directed)

    pair_paths_weight = torch.zeros((num_pair, 2, 2, num_rules), dtype=torch.float)
    for idx_path in range(num_paths):
        pair_paths_weight[i][1][0][paths_type[idx_path]] += paths_weight[idx_path]
        pair_paths_weight[i][0][1][paths_type_inv[idx_path]] += paths_weight_inv[idx_path]

    with TimerGuard('process edge_index', 'utils'):
        edge_index = torch.cat([data.edge_index.cpu(), data.edge_index[:, indices].cpu()[[1, 0], :], ], dim=-1)
        if directed:
            edge_type = torch.cat([edge_type, edge_type[indices]+num_edge_type], dim=-1)
            edge_degree = torch.cat([degree[2, data.node_id[data.edge_index[0, :]]], degree[1, data.node_id[data.edge_index[1, :]]][indices]], dim=-1)
        else:
            edge_type = torch.cat([edge_type, edge_type], dim=-1)
            edge_degree = degree[:, data.node_id[data.edge_index[0, :]]]
            edge_degree = edge_degree.repeat(2)
        edge_attrs = torch.stack([edge_type, edge_degree])
        edge_attrs_np = edge_attrs.numpy().transpose().copy()
        edge_attrs_np = edge_attrs_np.view(dtype).transpose().squeeze().copy()
    #with TimerGuard('to_networkx', 'utils'):
    #    G_undirected = to_networkx(Data(torch.zeros_like(data.node_id), edge_index, edge_type=edge_type.numpy(), edge_degree=edge_degree.numpy()), edge_attrs=['edge_type', 'edge_degree'], to_undirected=False)
    with TimerGuard('to_networkx', 'utils'):
        G_undirected = to_networkx(Data(torch.zeros_like(data.node_id), edge_index, edge_attrs=edge_attrs_np), edge_attrs=['edge_attrs'], to_undirected=False)
    with TimerGuard('all_simple_paths', 'utils'):
        paths2 = list(nx.all_simple_paths(G_undirected, source=src_idx, target=dst_idx, cutoff=max_len_rule))
        paths2 = sorted(paths2)
        paths_inv2 = cp.deepcopy(paths2)
        for path in paths_inv2:
            path.reverse()
    num_paths2 = len(paths2)
    #print(f'num paths {num_paths}')
    paths_edge_types2, paths_type2, paths_weight2, paths_degrees2, paths_gravity2 = process_paths2(paths2, data.node_id, G_undirected, degree, num_edge_type, directed)
    paths_edge_types_inv2, paths_type_inv2, paths_weight_inv2, paths_degrees_inv2, paths_gravity_inv2 = process_paths2(paths_inv2, data.node_id, G_undirected, degree, num_edge_type, directed)

    pair_paths_weight2 = torch.zeros((num_pair, 2, 2, num_rules), dtype=torch.float)
    for idx_path in range(num_paths2):
        pair_paths_weight2[i][1][0][paths_type2[idx_path]] += paths_weight2[idx_path]
        pair_paths_weight2[i][0][1][paths_type_inv2[idx_path]] += paths_weight_inv2[idx_path]
    
    print(num_paths == num_paths2)
    print(paths == paths2)
    print(paths_edge_types == paths_edge_types2)
    print(paths_type == paths_type2)
    print(np.where(np.array(paths_weight) - np.array(paths_weight2) > 1e-7))
    print(paths_degrees == paths_degrees2)
    print(paths_gravity == paths_gravity2)
    print(paths_edge_types_inv == paths_edge_types_inv2)
    print(paths_type_inv == paths_type_inv2)
    print(np.where(np.array(paths_weight_inv) - np.array(paths_weight_inv2) > 1e-7))
    print(paths_degrees_inv == paths_degrees_inv2)
    diff = [paths_degrees_inv[i]==paths_degrees_inv2[i] for i in range(num_paths)]
    print(paths_gravity_inv == paths_gravity_inv2)
    print(np.where(pair_paths_weight - pair_paths_weight2 > 1e-7))

    preprocess(data, degree, directed=directed,
                        use_len_spd=use_len_spd,
                        use_num_spd=use_num_spd,
                        use_cnb_jac=use_cnb_jac,
                        use_cnb_aa=use_cnb_aa,
                        use_cnb_ra=use_cnb_ra,
                        use_degree=use_degree,
                        gravity_type=gravity_type,
    )
    print(np.where(pair_paths_weight - data.pair_paths_weight > 1e-7))
    return

if __name__ == "__main__":
    data = torch.load('data.pt')
    degree = data.degree

    main(data, degree)
