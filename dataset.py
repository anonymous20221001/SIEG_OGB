#!/usr/bin/python
#****************************************************************#
# ScriptName: test_dataloader.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2022-02-20 11:38
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2022-03-15 12:07
# Function: 
#***************************************************************#

import os, sys
import pdb
from copy import deepcopy

from typing import Optional, Callable, List, Union, Tuple, Dict
from itertools import repeat, product
import numpy as np
import scipy.sparse as ssp
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_sparse import coalesce

from utils import *
from timer_guard import TimerGuard

# mock data, for test
uniq_x = torch.Tensor([[0], [0]]).long()
uniq_edge_index = torch.zeros((2,32)).long()
uniq_data = Data(x=uniq_x, edge_index=uniq_edge_index)

class SEALIterableDataset(IterableDataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False, **kwargs):
        self.root = root
        self.data = data
        #self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        self.sample_type = kwargs["sample_type"] if "sample_type" in kwargs else 0
        self.slice_type = kwargs["slice_type"] if "slice_type" in kwargs else 0
        self.shuffle = kwargs["shuffle"] if "shuffle" in kwargs else False
        self.preprocess_fn = kwargs["preprocess_fn"] if "preprocess_fn" in kwargs else None
        self.sizes = torch.Tensor([30 if max_nodes_per_hop is None else max_nodes_per_hop for i in range(num_hops)]).long()
        super(SEALIterableDataset, self).__init__()

        pos_edge, neg_edge = get_pos_neg_edges(split, split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes,
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t()
        self.link_nodes = torch.unique(self.links)
        self.labels = torch.Tensor([1] * pos_edge.size(1) + [0] * neg_edge.size(1)).long()

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.sample_type == 0:
            if self.directed:
                self.A_csc = self.A.tocsc()
            else:
                self.A_csc = None

        # for k_hop_subgraph_tensor
        elif self.sample_type == 1:
            self.adj = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1],
                                    value=edge_weight,
                                    sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
            self.adj_t = self.adj.t()
            self.adj.storage.rowptr()
            if self.directed:
                self.adj_t.storage.rowptr()
            else:
                self.adj_t = None

        # for k_hop_subgraph_sampler_tensor
        elif self.sample_type == 2:
            value = torch.arange(self.data.edge_index.size(1))
            self.adj_idc = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1],
                                        value=value,
                                        sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
            self.adj_idc_t = self.adj_idc.t()
            self.__val__ = edge_weight
            self.adj_idc.storage.rowptr()
            if self.directed:
                self.adj_idc_t.storage.rowptr()
            else:
                self.adj_idc_t = None

        self.pos_num_sample = pos_edge.size()[1]
        self.neg_num_sample = neg_edge.size()[1]
        self.num_sample = self.pos_num_sample + self.neg_num_sample
        if self.slice_type == 0:
            # slice with static number of samples in each group
            self.num_sample_in_slice = 1024
            self.pos_num_slice = math.ceil(self.pos_num_sample / self.num_sample_in_slice)
            self.neg_num_slice = math.ceil(self.neg_num_sample / self.num_sample_in_slice)
            self.num_slice = self.pos_num_slice + self.neg_num_slice
            self.pos_slice_pts = np.array([i * self.num_sample_in_slice for i in range(self.pos_num_slice)] + [self.pos_num_sample])
            self.neg_slice_pts = np.array([i * self.num_sample_in_slice for i in range(self.neg_num_slice)] + [self.neg_num_sample])
            self.slice_pts = np.concatenate([self.pos_slice_pts[:-1], self.neg_slice_pts + self.pos_num_sample], axis=0)
            print(f'num pos slice {self.pos_num_slice}')
            print(f'num neg slice {self.neg_num_slice}')
        else:
            # slice with static number of groups
            self.num_slice = 1024 * 4
            self.slice_pts = np.array([int(i * self.num_sample / self.num_slice) for i in range(self.num_slice)] + [self.num_sample])
        print(f'num pos sample {pos_edge.size()[1]}')
        print(f'num neg sample {neg_edge.size()[1]}')
        print(f'num sample {self.num_sample}')
        print(f'num slice {self.num_slice}')
        
        self.dirname = os.path.join(self.root, self.processed_directory_names)
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = next(iter(self))
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    @property
    def num_features(self) -> int:
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        data = next(iter(self))
        if hasattr(data, 'num_edge_features'):
            return data.num_edge_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_edge_features'")

    def __len__(self):
        return self.num_sample

    def len(self):
        return self.__len__()

    @property
    def processed_directory_names(self):
        name = 'SEAL_{}_data_sampler{}'.format(self.split, self.sample_type)
        if self.sample_type != 0:
            name += '_' + '-'.join([str(e) for e in self.sizes.tolist()])
        if self.percent != 100:
            name += f'_{self.percent}per'
        name += f'_{self.num_slice}slices'
        return name

    def __iter__(self):
        num_sample = self.num_sample
        num_slice = self.num_slice
        slice_pts = self.slice_pts
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            num_workers = 1
            worker_id = 0
        else:  # in a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        print(f'worker {worker_id}/{num_workers}, pid {os.getpid()}')
        worker_pts = np.array([int(i * num_slice / num_workers) for i in range(num_workers)] + [num_slice])
        #print('slice {}-{} in worker {}'.format(worker_pts[worker_id], worker_pts[worker_id+1], worker_id))
        for slice_id in range(worker_pts[worker_id], worker_pts[worker_id+1]):
            #print(slice_id, len(slice_pts), slice_pts[slice_id])
            basename = 'slice{}_in_{}(sample{}-{}).pt'.format(str(slice_id).zfill(len(str(num_slice))), num_slice, slice_pts[slice_id], slice_pts[slice_id+1]-1)
            filename = os.path.join(self.dirname, basename)
            if os.path.exists(filename):
                #print(f'load ${filename}')
                data_list, slices_list = torch.load(filename)
                #print('sample {}-{} in slice {}'.format(slice_pts[slice_id], slice_pts[slice_id+1], slice_id))
                i_list = np.array([i for i in range(slice_pts[slice_id + 1] - slice_pts[slice_id])])
                if self.shuffle:
                    np.random.seed()
                    perm = np.random.permutation(len(i_list))
                    i_list = np.array(i_list)[perm].tolist()
                for i in i_list:
                    data = Data()
                    for key in data_list.keys:
                        item, slices = data_list[key], slices_list[key]
                        start, end = slices[i].item(), slices[i + 1].item()
                        if torch.is_tensor(item):
                            s = list(repeat(slice(None), item.dim()))
                            cat_dim = data_list.__cat_dim__(key, item)
                            if cat_dim is None:
                                cat_dim = 0
                            s[cat_dim] = slice(start, end)
                        elif start + 1 == end:
                            s = slices[start]
                        else:
                            s = slice(start, end)
                        data[key] = item[s]
                    data.x = None if self.data.x is None else self.data.x[data.node_id, :]
                    if self.preprocess_fn is not None:
                        self.preprocess_fn(data, directed=self.directed)
                    yield data
            else:
                data_list = []
                # print('sample {}-{} in slice {}'.format(slice_pts[slice_id], slice_pts[slice_id+1], slice_id))
                idc = [idx for idx in range(slice_pts[slice_id], slice_pts[slice_id+1])]
                # if self.shuffle:
                #     np.random.seed(123)
                #     perm = np.random.permutation(len(idc))
                #     idc = np.array(idc)[perm].tolist()
                for idx in idc:
                    src, dst = self.links[idx].tolist()
                    y = self.labels[idx].tolist()
                    if self.sample_type == 0:
                        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                                             self.max_nodes_per_hop, node_features=self.data.x,
                                             y=y, directed=self.directed, A_csc=self.A_csc)
                    elif self.sample_type == 1:
                        tmp = k_hop_subgraph_tensor(src, dst, self.num_hops, self.A, self.adj, self.ratio_per_hop,
                                             self.max_nodes_per_hop, node_features=self.data.x,
                                             y=y, directed=self.directed, A_t=self.adj_t)
                    elif self.sample_type == 2:
                        tmp = k_hop_subgraph_sampler_tensor(src, dst, self.sizes, self.A, self.adj_idc,
                                             self.__val__,
                                             node_features=self.data.x,
                                             y=y, directed=self.directed, A_t=self.adj_idc_t)
                    data = construct_pyg_graph(*tmp, self.node_label)
                    data_copy = deepcopy(data)
                    if self.preprocess_fn is not None:
                        self.preprocess_fn(data, directed=self.directed)
                    yield data
                    del data_copy.x
                    data_list.append(data_copy)

                if data_list == []:
                    continue
                #print(f'save ${filename}')
                torch.save(self.collate(data_list), filename)

    @staticmethod
    def collate(data_list: List[Data]) -> Tuple[Data, Dict[str, torch.Tensor]]:
        r"""Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`."""
        keys = data_list[0].keys
        data = data_list[0].__class__()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if isinstance(item[key], torch.Tensor) and item[key].dim() > 0:
                cat_dim = item.__cat_dim__(key, item[key])
                cat_dim = 0 if cat_dim is None else cat_dim
                s = slices[key][-1] + item[key].size(cat_dim)
            else:
                s = slices[key][-1] + 1
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            item = data_list[0][key]
            if isinstance(item, torch.Tensor) and len(data_list) > 1:
                if item.dim() > 0:
                    cat_dim = data.__cat_dim__(key, item)
                    cat_dim = 0 if cat_dim is None else cat_dim
                    data[key] = torch.cat(data[key], dim=cat_dim)
                else:
                    data[key] = torch.stack(data[key])
            elif isinstance(item, torch.Tensor):  # Don't duplicate attributes...
                data[key] = data[key][0]
            elif isinstance(item, int) or isinstance(item, float):
                data[key] = torch.tensor(data[key])

            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices


class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False, sample_type=0, **kwargs):
        self.root = root
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        self.sample_type = kwargs["sample_type"] if "sample_type" in kwargs else 0
        self.preprocess_fn = kwargs["preprocess_fn"] if "preprocess_fn" in kwargs else None
        self.sizes = torch.Tensor([30 if max_nodes_per_hop is None else max_nodes_per_hop for i in range(num_hops)]).long()
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data_sizes{}'.format(self.split, '-'.join([str(e) for e in self.sizes.tolist()]))
        else:
            name = 'SEAL_{}_data_sizes{}_{}'.format(self.split, '-'.join([str(e) for e in self.sizes.tolist()]), self.percent)
        name += '.pt'
        return [name]

    def process(self):
        neg_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge,
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.sample_type == 0:
            if self.directed:
                A_csc = A.tocsc()
            else:
                A_csc = None

            # Extract enclosing subgraphs for pos and neg edges
            pos_list = extract_enclosing_subgraphs(
                pos_edge, A, self.data.x, 1, self.num_hops, self.node_label, 
                self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
            neg_list = extract_enclosing_subgraphs(
                neg_edge, A, self.data.x, 0, self.num_hops, self.node_label, 
                self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        else:
            # for k_hop_subgraph_sampler_tensor
            value = torch.arange(self.data.edge_index.size(1))
            self.adj_idc = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1],
                                      value=value,
                                      sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
            self.adj_idc_t = self.adj_idc.t()
            self.__val__ = edge_weight
            self.adj_idc.storage.rowptr()
            if self.directed:
                self.adj_idc_t.storage.rowptr()
            else:
                self.adj_idc_t = None

            pos_list = extract_enclosing_subgraphs_tensor(
                pos_edge, A, self.adj_idc, self.sizes, self.__val__, self.data.x, 1,
                self.num_hops, self.node_label, self.directed, self.adj_idc_t)
            neg_list = extract_enclosing_subgraphs_tensor(
                neg_edge, A, self.adj_idc, self.sizes, self.__val__, self.data.x, 0,
                self.num_hops, self.node_label, self.directed, self.adj_idc_t)

        data_list = pos_list + neg_list
        if self.preprocess_fn is not None:
            for data in tqdm(data_list):
                self.preprocess_fn(data, directed=self.directed)
        torch.save(self.collate(data_list), self.processed_paths[0])
        del pos_list, neg_list, data_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False, sample_type=0, **kwargs):
        self.root = root
        self.data = data
        #self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        self.sample_type = kwargs["sample_type"] if "sample_type" in kwargs else 0
        self.preprocess_fn = kwargs["preprocess_fn"] if "preprocess_fn" in kwargs else None
        super(SEALDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t()
        self.link_nodes = torch.unique(self.links)
        self.labels = torch.Tensor([1] * pos_edge.size(1) + [0] * neg_edge.size(1)).long()

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.sample_type == 0:
            if self.directed:
                self.A_csc = self.A.tocsc()
            else:
                self.A_csc = None

        # for k_hop_subgraph_tensor
        elif self.sample_type == 1:
            self.adj = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1],
                                      value=edge_weight,
                                      sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
            self.adj_t = self.adj.t()
            self.adj.storage.rowptr()
            if self.directed:
                self.adj_t.storage.rowptr()
            else:
                self.adj_t = None

        # for k_hop_subgraph_sampler_tensor
        elif self.sample_type == 2:
            value = torch.arange(self.data.edge_index.size(1))
            self.adj_idc = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1],
                                      value=value,
                                      sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
            self.adj_idc_t = self.adj_idc.t()
            self.__val__ = edge_weight
            self.adj_idc.storage.rowptr()
            if self.directed:
                self.adj_idc_t.storage.rowptr()
            else:
                self.adj_idc_t = None

        self.sizes = torch.Tensor([30 if max_nodes_per_hop is None else max_nodes_per_hop for i in range(num_hops)]).long()

    def __len__(self):
        return self.links.size()[0]

    def len(self):
        return self.__len__()

    @TimerGuard('get', 'utils')
    def get(self, idx):
        src, dst = self.links[idx].tolist()
        y = self.labels[idx].tolist()
        if self.sample_type == 0:
            tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop, 
                                 self.max_nodes_per_hop, node_features=self.data.x, 
                                 y=y, directed=self.directed, A_csc=self.A_csc)
        elif self.sample_type == 1:
            tmp = k_hop_subgraph_tensor(src, dst, self.num_hops, self.A, self.adj, self.ratio_per_hop, 
                                 self.max_nodes_per_hop, node_features=self.data.x, 
                                 y=y, directed=self.directed, A_t=self.adj_t)
        elif self.sample_type == 2:
            tmp = k_hop_subgraph_sampler_tensor(src, dst, self.sizes, self.A, self.adj_idc,
                                 self.__val__,
                                 node_features=self.data.x, 
                                 y=y, directed=self.directed, A_t=self.adj_idc_t)
        #_, subgraph, _, _, _ = tmp
        #print(f"subgraph: num_nodes {subgraph.shape[0]}, num_edges {subgraph.size}")
        data = construct_pyg_graph(*tmp, self.node_label)
        if self.preprocess_fn is not None:
            self.preprocess_fn(data, directed=self.directed)

        return data

