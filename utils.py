# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
import torch_sparse
from torch_sparse import spspmm, SparseTensor
import torch_geometric
from torch_geometric.data.sampler import Adj, EdgeIndex
from torch_geometric.data import DataLoader as PygDataLoader
from torch_geometric.data import Data as PygData
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
from timer_guard import TimerGuard
from collection import Collections
import copy as cp
import pdb


# @TimerGuard("neighbors", "utils")
def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


#@TimerGuard("neighbors_tensor", "utils")
def neighbors_tensor(fringe: torch.Tensor, A:torch_sparse.SparseTensor, outgoing=True) -> torch.Tensor:
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    idc = fringe.tolist()
    res_list = []
    edge_index_list = []
    val_list = []
    if outgoing:
        for idx in idc:
            A_select = A[idx]
            row, col, val = A_select.coo()
            res_list.append(col)
            edge_index_list.append(torch.vstack([fringe[row], col]))
            val_list.append(val)
    else:
        for idx in idc:
            A_select = A.index_select(1, idx)
            row, col, val = A_select.coo()
            res_list.append(row)
            edge_index_list.append(torch.vstack([row, fringe[col]]))
            val_list.append(val)
    res = torch.cat(res_list).unique()
    edge_index = torch.cat(edge_index_list, dim=1)
    val = torch.cat(val_list)

    return res, edge_index, val


#@TimerGuard("enclose_index", "utils")
def enclose_index(A_ssp: ssp.csr_matrix, nodes: list):
    try:
        A_ssp_half = A_ssp[nodes].tocoo()
        data = A_ssp_half.data
        row = A_ssp_half.row
        col = A_ssp_half.col
        nodes = torch.Tensor(nodes).long()
        keep = (torch.from_numpy(col) == nodes.view(-1, 1)).any(dim=0).numpy()
        num_nodes = nodes.numel()
        data, row, col = data[keep], row[keep], cp.deepcopy(col[keep])
        #_, col = Collections.re_mapping(nodes, torch.from_numpy(col).long())
        #col = col.numpy()
        mapper = {x.item(): i for i, x in enumerate(nodes)}
        col = np.array([mapper[x.item()] for x in col], dtype=np.int32)
        subgraph = ssp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    except:
        print(nodes)
        print(data, row, col)
    return subgraph


#@TimerGuard("k_hop_subgraph", "utils")
def k_hop_subgraph(src: int, dst: int, num_hops: int, A: ssp.csr_matrix, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features=None, 
                   y=1, directed=False, A_csc: ssp.csc_matrix=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            max_nodes_this_hop = max_nodes_per_hop[dist-1] if isinstance(max_nodes_per_hop, list) else max_nodes_per_hop
            if max_nodes_this_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_this_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    # Enclose subgraph
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


#@TimerGuard("k_hop_subgraph_tensor", "utils")
def k_hop_subgraph_tensor(src: int, dst: int, num_hops: int, A_ssp: ssp.csr_matrix, A: torch_sparse.SparseTensor, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features: torch.Tensor=None, 
                   y: int=1, directed=False, A_t: torch_sparse.SparseTensor=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = torch.Tensor([src, dst]).long()
    dists = torch.Tensor([0, 0]).long()
    visited = torch.Tensor([src, dst]).long().unique()
    fringe = torch.Tensor([src, dst]).long().unique()
    edge_index_list = []
    val_list = []
    for dist in range(1, num_hops+1):
        if not directed:
            fringe, edge_index, val = neighbors_tensor(fringe, A)
            edge_index_list.append(edge_index)
            val_list.append(val)
        else:
            out_neighbors, out_edge_index, out_val = neighbors_tensor(fringe, A)
            in_neighbors, in_edge_index, in_val = neighbors_tensor(fringe, A_t)
            fringe = Collections.set_union(out_neighbors, in_neighbors)
            edge_index = torch.cat([out_edge_index, in_edge_index[[1, 0]]], dim=1)
            edge_index_list.append(edge_index)
            val = torch.cat([out_val, in_val])
            val_list.append(val)
        fringe = Collections.set_difference(fringe, visited)
        visited = Collections.set_union(visited, fringe)

        if sample_ratio < 1.0:
            indices = torch.randperm(fringe.numel())[:int(sample_ratio*len(fringe))]
            fringe = fringe[indices]
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                indices = torch.randperm(fringe.numel())[:max_nodes_per_hop]
                fringe = fringe[indices]
        if fringe.numel() == 0:
            break
        nodes = torch.cat([nodes, fringe])
        dists = torch.cat([dists, torch.zeros((len(fringe)), dtype=torch.long) + dist])
    #edge_index = torch.cat(edge_index_list, dim=1)
    #val = torch.cat(val_list)
    #comp = torch.stack([(edge_index[0].t().unsqueeze(axis=1) == nodes.unsqueeze(axis=1).t()), (edge_index[1].t().unsqueeze(axis=1) == nodes.unsqueeze(axis=1).t())])
    #valid_idc = comp.any(-1).all(0)
    #edge_index_val = torch.cat([edge_index, val.unsqueeze(0)], dim=0)
    #edge_index_val = edge_index_val[:, valid_idc]
    #edge_index_val = torch.unique(edge_index_val, dim=1)
    #edge_index = edge_index_val[:2, :]
    #val = edge_index_val[2, :]
    #mapper = {x.item(): i for i, x in enumerate(nodes)}
    #edge_index = torch.tensor([mapper[x.item()] for x in edge_index.flatten()]).reshape(2, -1)
    #num_nodes = nodes.numel()
    #num_edges = edge_index.size(1)
    ##subgraph = torch_sparse.cat(A[nodes], A_t[nodes])
    ##subgraph = subgraph[:, nodes]
    ##subgraph = subgraph.to_scipy(layout='csr')
    #subgraph = ssp.csr_matrix((val, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    nodes = nodes.tolist()
    dists = dists.tolist()

    subgraph = enclose_index(A_ssp, nodes)

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


#@TimerGuard("k_hop_subgraph_sampler_tensor", "utils")
def k_hop_subgraph_sampler_tensor(src: int, dst: int, sizes, A_ssp: ssp.csr_matrix, A: torch_sparse.SparseTensor, 
                   val: torch.Tensor, node_features=None, 
                   y: int=1, directed=False, A_t: torch_sparse.SparseTensor=None):
    num_hops = len(sizes)
    nodes = torch.Tensor([src, dst]).long()
    dists = torch.Tensor([0, 0]).long()
    visited = torch.unique(torch.Tensor([src, dst]).long())
    fringe = torch.unique(torch.Tensor([src, dst]).long())
    for dist in range(num_hops):
        size = sizes[dist]
        n_id_list = []

        _, n_id = A.sample_adj(fringe, size, replace=False)
        n_id_list.append(n_id)

        if directed:
            _, n_id = A_t.sample_adj(fringe, size, replace=False)
            n_id_list.append(n_id)

        fringe = torch.unique(torch.cat(n_id_list))
        fringe = Collections.set_difference(fringe, visited)
        visited = Collections.set_union(visited, fringe)
        max_nodes_per_hop = sizes[dist].tolist()
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                indices = torch.randperm(fringe.numel())[:max_nodes_per_hop]
                fringe = fringe[indices]
        if fringe.numel() == 0:
            break

        nodes = torch.cat([nodes, fringe])
        dists = torch.cat([dists, torch.ones((fringe.numel()), dtype=torch.long) + dist])

    nodes = nodes.tolist()
    dists = dists.tolist()

    subgraph = enclose_index(A_ssp, nodes)

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


#@TimerGuard("drnl_node_labeling", "utils")
def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    # dist_over_2, dist_mod_2 = dist // 2, dist % 2
    dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More 
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


#@TimerGuard("construct_pyg_graph", "utils")
def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl'):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists)==0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z>100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = PygData(node_features, edge_index, edge_weight=edge_weight, y=y, z=z, 
                node_id=node_ids, num_nodes=num_nodes)
    #data = PygData(node_features, edge_index, edge_weight=edge_weight, y=y, z=z, 
    #            node_id=node_ids, num_nodes=num_nodes, adj=adj)
    return data

 
def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl', 
                                ratio_per_hop=1.0, max_nodes_per_hop=None, 
                                directed=False, A_csc=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop, 
                             max_nodes_per_hop, node_features=x, y=y, 
                             directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(*tmp, node_label)
        data_list.append(data)

    return data_list


def extract_enclosing_subgraphs_tensor(link_index, A_ssp: ssp.csr_matrix, A: torch_sparse.SparseTensor,
                                sizes: torch.Tensor, val: torch.Tensor,
                                x: torch.Tensor, y: int, num_hops: int, node_label='drnl', 
                                directed=False, A_t: torch_sparse.SparseTensor=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        tmp = k_hop_subgraph_sampler_tensor(src, dst, sizes, A_ssp, A, 
                           val, node_features=x, 
                           y=y, directed=directed, A_t=A_t)
        data = construct_pyg_graph(*tmp, node_label)
        data_list.append(data)

    return data_list


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1, neg_ratio=1):
    data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        # edge_index和data.train_neg_edge_index后面没用，在get_pos_neg_edges里重新计算了这俩
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=int(data.train_pos_edge_index.size(1)*neg_ratio))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=int(row.size(0)*neg_ratio))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def get_dict_info(d):
    info = ''
    for k,v in d.items():
        if isinstance(v, torch.Tensor):
            info += '{}: {}\n'.format(k, v.size())
        elif isinstance(v, np.ndarray):
            info += '{}: {}\n'.format(k, v.shape)
        elif isinstance(v, list):
            info += '{}: {}\n'.format(k, len(v))
        elif isinstance(v, dict):
            info += '{}:\n{}'.format(k, get_dict_info(v))
    return info


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100, neg_ratio=1):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        # if split == 'train':  # 非ogbl训练集本就有edge_neg，会重采样，应该不影响PubMed/Cora/CiteSeer结果
        if 'edge_neg' not in split_edge[split]:
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=int(pos_edge.size(1)*neg_ratio))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), int(neg_ratio)],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        if percent < 100:
            np.random.seed(123)
            num_source = source.size(0)
            perm = np.random.permutation(num_source)
            perm = perm[:int(percent / 100 * num_source)]
            source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def np_sampling(rw_dict, ptr, neighs, bsize, target, num_walks=100, num_steps=3, nthread=-1):
    with tqdm(total=len(target)) as pbar:
        for batch in gen_batch(target, bsize, True):
            walk_set, freqs = run_walk(ptr, neighs, batch, num_walks=num_walks, num_steps=num_steps, replacement=True, nthread=nthread)
            node_id, node_freq = freqs[:, 0], freqs[:, 1]
            rw_dict.update(dict(zip(batch, zip(walk_set, node_id, node_freq))))
            pbar.update(len(batch))
    return rw_dict


def gen_dataset(dataset, graphs, args, bsize=10000, nthread=-1):
    G_val, G_full = graphs['val'], graphs['test']

    keep_neg = False if 'ppa' not in args.dataset else True
    #keep_neg = False

    test_pos_edge, test_neg_edge = get_pos_neg_edges('test', dataset.split_edge, ratio=args.test_ratio,
                                                     keep_neg=keep_neg)
    val_pos_edge, val_neg_edge = get_pos_neg_edges('valid', dataset.split_edge, ratio=args.valid_ratio,
                                                   keep_neg=keep_neg)

    val_dict = test_dict = np_sampling({}, G_val.indptr, G_val.indices, bsize=bsize,
                                       target=torch.unique(
                                           torch.cat([inf_set['val']['E'], inf_set['test']['E']])).tolist(),
                                       num_walks=args.num_walk, num_steps=args.num_step - 1, nthread=nthread)

    args.w_max = dataset.train_wmax if args.use_weight else None

    return test_dict, val_dict, inf_set


def CN(A, edge_index, batch_size=100000, cn_types=['in']):
    # The Common Neighbor heuristic score.
    num_nodes = A.shape[0]
    A_t = A.transpose().tocsr()

    x_ind, y_ind = A.nonzero()
    weights = np.array(A[x_ind, y_ind]).flatten()
    if (A.todense() == A_t.todense()).all():
        A_undirected = A
    else:
        A_undirected = ssp.csr_matrix((np.concatenate([weights, weights]), (np.concatenate([x_ind, y_ind]), np.concatenate([y_ind, x_ind]))), shape=(num_nodes, num_nodes))

    link_loader = PygDataLoader(range(edge_index.size(1)), batch_size)
    multi_type_scores = []
    for cn_type in cn_types:
        scores = []
        for ind in tqdm(link_loader):
            src, dst = edge_index[0, ind], edge_index[1, ind]
            if cn_type == 'undirected':
                # undirected
                cur_scores = np.array(np.sum(A_undirected[src].multiply(A_undirected[dst]), 1)).flatten()
            elif cn_type == 'in':
                # center in
                cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
            elif cn_type == 'out':
                # center out
                cur_scores = np.array(np.sum(A_t[src].multiply(A_t[dst]), 1)).flatten()
            elif cn_type == 's2o':
                # source to destination
                cur_scores = np.array(np.sum(A[src].multiply(A_t[dst]), 1)).flatten()
            elif cn_type == 'o2s':
                # destination to source
                cur_scores = np.array(np.sum(A_t[src].multiply(A[dst]), 1)).flatten()
            scores.append(cur_scores)
        multi_type_scores.append(torch.FloatTensor(np.concatenate(scores, 0)))
    return torch.stack(multi_type_scores), edge_index


def Jaccard(A, edge_index, batch_size=100000, cn_types=['in']):
    # The Adamic-Adar heuristic score.
    num_nodes = A.shape[0]
    degree_in = A.sum(axis=0).getA1() # in_degree

    A_t = A.transpose().tocsr()
    degree_out = A_t.sum(axis=0).getA1()

    x_ind, y_ind = A.nonzero()
    weights = np.array(A[x_ind, y_ind]).flatten()
    if (A.todense() == A_t.todense()).all():
        A_undirected = A
    else:
        A_undirected = ssp.csr_matrix((np.concatenate([weights, weights]), (np.concatenate([x_ind, y_ind]), np.concatenate([y_ind, x_ind]))), shape=(num_nodes, num_nodes))
    degree_undirected = A_undirected.sum(axis=0).getA1() # degree

    link_loader = PygDataLoader(range(edge_index.size(1)), batch_size)
    multi_type_scores = []
    for cn_type in cn_types:
        scores = []
        for ind in tqdm(link_loader):
            src, dst = edge_index[0, ind], edge_index[1, ind]
            if cn_type == 'undirected':
                # undirected
                intersection_scores = np.array(np.sum(A_undirected[src].multiply(A_undirected[dst]), 1)).flatten()
                union_scores = degree_undirected[src] + degree_undirected[dst] - intersection_scores
            elif cn_type == 'in':
                # center in
                intersection_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
                union_scores = degree_out[src] + degree_out[dst] - intersection_scores
            elif cn_type == 'out':
                # center out
                intersection_scores = np.array(np.sum(A_t[src].multiply(A_t[dst]), 1)).flatten()
                union_scores = degree_in[src] + degree_in[dst] - intersection_scores
            elif cn_type == 's2o':
                # source to destination
                intersection_scores = np.array(np.sum(A[src].multiply(A_t[dst]), 1)).flatten()
                union_scores = degree_out[src] + degree_in[dst] - intersection_scores
            elif cn_type == 'o2s':
                # destination to source
                intersection_scores = np.array(np.sum(A_t[src].multiply(A[dst]), 1)).flatten()
                union_scores = degree_in[src] + degree_out[dst] - intersection_scores
            cur_scores = intersection_scores / union_scores
            cur_scores[np.isinf(cur_scores)] = 0
            cur_scores[np.isnan(cur_scores)] = 0
            scores.append(cur_scores)
        multi_type_scores.append(torch.FloatTensor(np.concatenate(scores, 0)))
    return torch.stack(multi_type_scores), edge_index


def AA(A, edge_index, batch_size=100000, cn_types=['in']):
    # The Adamic-Adar heuristic score.
    num_nodes = A.shape[0]
    div_log_deg_multiplier = 1 / np.log(A.sum(axis=0)) # in_degree
    div_log_deg_multiplier[np.isinf(div_log_deg_multiplier)] = 2.0
    A_div_log_deg = A.multiply(div_log_deg_multiplier).tocsr()

    A_t = A.transpose().tocsr()
    div_log_deg_multiplier_t = 1 / np.log(A_t.sum(axis=0))
    div_log_deg_multiplier_t[np.isinf(div_log_deg_multiplier_t)] = 2.0
    A_t_div_log_deg = A_t.multiply(div_log_deg_multiplier_t).tocsr()

    x_ind, y_ind = A.nonzero()
    weights = np.array(A[x_ind, y_ind]).flatten()
    if (A.todense() == A_t.todense()).all():
        A_undirected = A
    else:
        A_undirected = ssp.csr_matrix((np.concatenate([weights, weights]), (np.concatenate([x_ind, y_ind]), np.concatenate([y_ind, x_ind]))), shape=(num_nodes, num_nodes))
    div_log_deg_multiplier_undirected = 1 / np.log(A_undirected.sum(axis=0)) # degree
    div_log_deg_multiplier_undirected[np.isinf(div_log_deg_multiplier_undirected)] = 2.0
    if 'undirected' in cn_types:
        A_undirected_div_log_deg = A_undirected.multiply(div_log_deg_multiplier_undirected).tocsr()
    A_t_undirected_div_log_deg = A_t.multiply(div_log_deg_multiplier_undirected).tocsr()

    link_loader = PygDataLoader(range(edge_index.size(1)), batch_size)
    multi_type_scores = []
    for cn_type in cn_types:
        scores = []
        for ind in tqdm(link_loader):
            src, dst = edge_index[0, ind], edge_index[1, ind]
            if cn_type == 'undirected':
                # undirected
                cur_scores = np.array(np.sum(A_undirected[src].multiply(A_undirected_div_log_deg[dst]), 1)).flatten()
            elif cn_type == 'in':
                # center in
                cur_scores = np.array(np.sum(A[src].multiply(A_div_log_deg[dst]), 1)).flatten()
            elif cn_type == 'out':
                # center out
                cur_scores = np.array(np.sum(A_t[src].multiply(A_t_div_log_deg[dst]), 1)).flatten()
            elif cn_type == 's2o':
                # source to destination
                cur_scores = np.array(np.sum(A[src].multiply(A_t_undirected_div_log_deg[dst]), 1)).flatten()
            elif cn_type == 'o2s':
                # destination to source
                cur_scores = np.array(np.sum(A_t_undirected_div_log_deg[src].multiply(A[dst]), 1)).flatten()
            scores.append(cur_scores)
        multi_type_scores.append(torch.FloatTensor(np.concatenate(scores, 0)))
    return torch.stack(multi_type_scores), edge_index


def RA(A, edge_index, batch_size=100000, cn_types=['in']):
    # The Adamic-Adar heuristic score.
    num_nodes = A.shape[0]
    div_deg_multiplier = 1 / A.sum(axis=0) # in_degree
    div_deg_multiplier[np.isinf(div_deg_multiplier)] = 0.0
    A_div_deg = A.multiply(div_deg_multiplier).tocsr()

    A_t = A.transpose().tocsr()
    div_deg_multiplier_t = 1 / A_t.sum(axis=0)
    div_deg_multiplier_t[np.isinf(div_deg_multiplier_t)] = 0.0
    A_t_div_deg = A_t.multiply(div_deg_multiplier_t).tocsr()

    x_ind, y_ind = A.nonzero()
    weights = np.array(A[x_ind, y_ind]).flatten()
    if (A.todense() == A_t.todense()).all():
        A_undirected = A
    else:
        A_undirected = ssp.csr_matrix((np.concatenate([weights, weights]), (np.concatenate([x_ind, y_ind]), np.concatenate([y_ind, x_ind]))), shape=(num_nodes, num_nodes))
    div_deg_multiplier_undirected = 1 / A_undirected.sum(axis=0) # degree
    div_deg_multiplier_undirected[np.isinf(div_deg_multiplier_undirected)] = 0.0
    if 'undirected' in cn_types:
        A_undirected_div_deg = A_undirected.multiply(div_deg_multiplier_undirected).tocsr()

    link_loader = PygDataLoader(range(edge_index.size(1)), batch_size)
    multi_type_scores = []
    for cn_type in cn_types:
        scores = []
        for ind in tqdm(link_loader):
            src, dst = edge_index[0, ind], edge_index[1, ind]
            if cn_type == 'undirected':
                # undirected
                cur_scores = np.array(np.sum(A_undirected[src].multiply(A_undirected_div_deg[dst]), 1)).flatten()
            elif cn_type == 'in':
                # center in
                cur_scores = np.array(np.sum(A[src].multiply(A_div_deg[dst]), 1)).flatten()
            elif cn_type == 'out':
                # center out
                cur_scores = np.array(np.sum(A_t[src].multiply(A_t_div_deg[dst]), 1)).flatten()
            elif cn_type == 's2o':
                # source to destination
                cur_scores = np.array(np.sum(A[src].multiply(A_t_div_deg[dst]), 1)).flatten()
            elif cn_type == 'o2s':
                # destination to source
                cur_scores = np.array(np.sum(A_t[src].multiply(A_div_deg[dst]), 1)).flatten()
            scores.append(cur_scores)
        multi_type_scores.append(torch.FloatTensor(np.concatenate(scores, 0)))
    return torch.stack(multi_type_scores), edge_index


def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    #edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.runs = runs
        self.results = [[] for _ in range(self.runs)]

    def reset(self):
        self.results = [[] for _ in range(self.runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout, std=False):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:-1, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:-1, 0].max():.2f}', file=f)
            print(f'Highest Eval Point: {argmax + 1}', file=f)
            print(f'Final valid: {result[-1, 0]:.2f}', file=f)
            print(f' Final Test: {result[-1, 1]:.2f}', file=f)
            if std:
                print(f'Highest Valid: {result[:, 0].mean():.2f} ± {result[:, 0].std():.2f}', file=f)
                print(f'   Final Test: {result[:, 1].mean():.2f} ± {result[:, 1].std():.2f}', file=f)
        else:
            results = []
            for e in self.results:
                if len(e) > 0:
                    results.append(e)
            result = 100 * torch.tensor(results)

            best_results = []
            for r in result:
                valid = r[-1, 0].item()
                test = r[-1, 1].item()
                # valid = r[:, 0].max().item()
                # test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Final Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f' Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)

