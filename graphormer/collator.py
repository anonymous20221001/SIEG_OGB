# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from torch_geometric.data import Batch
import pdb


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    if x.dim() == 1:
        return pad_1d_unsqueeze(x, padlen)
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_square_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def collator(data_list):
    xs = [data.x for data in data_list]
    zs = [data.z for data in data_list]
    node_ids = [data.node_id for data in data_list]
    edge_indexes = [data.edge_index for data in data_list]
    edge_weights = [data.edge_weight for data in data_list]
    ys = [data.y for data in data_list]

    max_node_num = max(i.size(0) for i in zs)
    num_batch = len(zs)

    x = torch.cat(xs)
    z = torch.cat(zs)
    node_id = torch.cat(node_ids)
    batch = torch.ones(x.shape[0], dtype=int)
    ptr = torch.ones(num_batch+1, dtype=int)
    ptr[0] = 0
    for b, e in enumerate(zs):
        ptr[b+1] = e.shape[0]
        batch[ptr[b]:ptr[b+1]] = ptr[b]
        edge_indexes[b] += ptr[b]
    edge_index = torch.cat(edge_indexes, axis=1)
    edge_weight = torch.cat(edge_weights)
    y = torch.cat(ys)
    batched_data = Batch(x=x, batch=batch, ptr=ptr, edge_index=edge_index, node_id=node_id, edge_weight=edge_weight, z=z, y=y)

    # pairwise structure
    batched_data.pair_x = torch.cat([pad_2d_unsqueeze(e, max_node_num) for e in xs])
    batched_data.pair_z = torch.cat([pad_2d_unsqueeze(e, max_node_num) for e in zs])
    batched_data.pair_edge_idx = edge_index
    pair_attn_biases = [data.pair_attn_bias for data in data_list]
    batched_data.pair_attn_bias = torch.cat([pad_attn_bias_unsqueeze(e[0], max_node_num + 1) for e in pair_attn_biases])

    if 'pair_len_shortest_path' in data_list[0].keys:
        pair_len_shortest_paths = [data.pair_len_shortest_path for data in data_list]
        batched_data.pair_len_shortest_path = torch.cat([pad_2d_square_unsqueeze(e[0], max_node_num) for e in pair_len_shortest_paths])
    if 'pair_num_shortest_path' in data_list[0].keys:
        pair_num_shortest_paths = [data.pair_num_shortest_path for data in data_list]
        batched_data.pair_num_shortest_path = torch.cat([pad_2d_square_unsqueeze(e[0], max_node_num) for e in pair_num_shortest_paths])
    if 'pair_undir_jac' in data_list[0].keys:
        pair_undir_jacs = [data.pair_undir_jac for data in data_list]
        batched_data.pair_undir_jac = torch.cat([pad_2d_square_unsqueeze(e[0], max_node_num) for e in pair_undir_jacs])
    if 'pair_undir_aa' in data_list[0].keys:
        pair_undir_aas = [data.pair_undir_aa for data in data_list]
        batched_data.pair_undir_aa = torch.cat([pad_2d_square_unsqueeze(e[0], max_node_num) for e in pair_undir_aas])
    if 'pair_undir_ra' in data_list[0].keys:
        pair_undir_ras = [data.pair_undir_ra for data in data_list]
        batched_data.pair_undir_ra = torch.cat([pad_2d_square_unsqueeze(e[0], max_node_num) for e in pair_undir_ras])
    if 'pair_undir_degree' in data_list[0].keys:
        pair_undir_degrees = [data.pair_undir_degree for data in data_list]
        batched_data.pair_undir_degrees = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in pair_undir_degrees])
    if 'pair_in_degree' in data_list[0].keys:
        pair_in_degrees = [data.pair_in_degree for data in data_list]
        batched_data.pair_in_degrees = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in pair_in_degrees])
    if 'pair_out_degree' in data_list[0].keys:
        pair_out_degrees = [data.pair_out_degree for data in data_list]
        batched_data.pair_out_degrees = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in pair_out_degrees])

    return batched_data