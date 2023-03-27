# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU,
                      Sequential, BatchNorm1d as BN)
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, SAGEConv, GINConv,
                                global_sort_pool, global_add_pool, global_mean_pool)
from torch_geometric.data import Data
from graphormer.model import Graphormer
import ngnn_models
import pdb


# x: (N, 128)，点特征，N是一个batch的所有节点数，每个子图节点数不同
# z: (N)，labeling trick里的DRNL，每个节点到头尾节点的距离（一维）
# z_embedding: (1000, 32)，z的lookup table
# edge_index: (2, Ne)
# batch: (N)，代表每个节点属于哪一个子图的标签，类似[0, .., 0, 1, ..., 1, 2, ..., 31]
# node_embedding: 对全局的node_id进行编码得到的embedding
# node_id: 全局点序号
# rpe：surel里的相对位置编码

class NGNN_GCNConv(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NGNN_GCNConv, self).__init__()
        self.conv = GCNConv(input_channels, hidden_channels)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, output_channels)

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x, edge_weight=None):
        x = self.conv(g, x, edge_weight)
        x = F.relu(x)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(GCN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            # self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim))
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            x = torch.cat([x, x_rpe_emb], 1)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)

        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset=None, 
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(SAGE, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))
        self.convs.append(SAGEConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            x = torch.cat([x, x_rpe_emb], 1)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None,
                 GNN=GCNConv, NGNN=NGNN_GCNConv, use_feature=False,
                 node_embedding=None):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                num_nodes = []
                for i, data in enumerate(iter(train_dataset)):
                    num_nodes += [data.num_nodes]
                    if i >= 1000:
                        break
                num_nodes = sorted(num_nodes)
                # _sampled_indices = list(range(1000)) + list(range(len(train_dataset) - 1000, len(train_dataset)))
                # num_nodes = sorted([train_dataset[i][0].num_nodes() for i in _sampled_indices])

                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)

        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            # initial_channels += self.rpe_hidden_dim * 2
            initial_channels += self.rpe_hidden_dim
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        if args.use_ignn:
            self.convs.append(NGNN(initial_channels, hidden_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(NGNN(hidden_channels, hidden_channels, hidden_channels))
            self.convs.append(NGNN(hidden_channels, hidden_channels, 1))
        else:
            self.convs.append(GNN(initial_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            # x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            x_rpe_emb = self.trainable_embedding(x_rpe).sum(dim=-2)
            x = torch.cat([x, x_rpe_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN_noNeigFeat(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None,
                 GNN=GCNConv, NGNN=NGNN_GCNConv, use_feature=False,
                 node_embedding=None):
        super(DGCNN_noNeigFeat, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                num_nodes = []
                for i, data in enumerate(iter(train_dataset)):
                    num_nodes += [data.num_nodes]
                    if i >= 1000:
                        break
                num_nodes = sorted(num_nodes)
                # _sampled_indices = list(range(1000)) + list(range(len(train_dataset) - 1000, len(train_dataset)))
                # num_nodes = sorted([train_dataset[i][0].num_nodes() for i in _sampled_indices])

                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)

        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels = hidden_channels
        assert train_dataset.num_features > 0
        feat_channels = train_dataset.num_features
        self.lin01 = Linear(feat_channels, hidden_channels)  # FFN单独编码feature
        self.lin02 = Linear(num_layers * hidden_channels + 1 + hidden_channels, num_layers * hidden_channels + 1)  # 3*32+1+32, 3*32+1

        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            # initial_channels += self.rpe_hidden_dim * 2
            initial_channels += self.rpe_hidden_dim
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        if args.use_ignn:
            self.convs.append(NGNN(initial_channels, hidden_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(NGNN(hidden_channels, hidden_channels, hidden_channels))
            self.convs.append(NGNN(hidden_channels, hidden_channels, 1))
        else:
            self.convs.append(GNN(initial_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        xs = [z_emb]
        ffn_feat = x.to(torch.float)

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # linear -> concat -> linear
        x_feature = F.relu(self.lin01(ffn_feat))  # FFN单独编码feature
        x = torch.cat((x, x_feature), dim=1)
        x = F.relu(self.lin02(x))

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

class GIN(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, 
                 jk=True, train_eps=False):
        super(GIN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))
        self.conv1 = GINConv(
            Sequential(
                Linear(initial_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ),
            train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ),
                    train_eps=train_eps))

        self.dropout = dropout
        if self.jk:
            self.lin1 = Linear(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            x = torch.cat([x, x_rpe_emb], 1)

        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.jk:
            x = global_mean_pool(torch.cat(xs, dim=1), batch)
        else:
            x = global_mean_pool(xs[-1], batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


def abstract_pair_data(data, z_emb_pair=None):
    if z_emb_pair is None:
        pair_data = Data(x=data.pair_x, z=data.pair_z, edge_index=data.pair_edge_idx)
    else:  # 传入z_emb，就用z_emb替代feature
        pair_data = Data(x=z_emb_pair, z=data.pair_z, edge_index=data.pair_edge_idx)
    for key in data.keys:
        if key.startswith('pair_') and key not in ['pair_x', 'pair_z', 'pair_edge_idx']:
            pair_data[key[5:]] = data[key]
    return pair_data

class GCNGraphormer(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, use_feature_GT=True, node_embedding=None, dropout=0.5):
        super(GCNGraphormer, self).__init__()
        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        # 不用feature，就用z_emb代替，维度就是hidden_channels
        input_dim = train_dataset.num_features if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=hidden_channels,
                                     ffn_dim=hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.graphormer.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            h = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            h = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            h = torch.cat([h, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            h = torch.cat([h, x_rpe_emb], 1)

        for conv in self.convs[:-1]:
            h = conv(h, edge_index, edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index, edge_weight)

        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            h_src = h[center_indices]
            h_dst = h[center_indices + 1]
            h = h_src * h_dst
            if self.use_feature_GT:
                pair_data = abstract_pair_data(data)
            else:
                z_emb_src = z_emb[center_indices]
                z_emb_dst = z_emb[center_indices + 1]
                z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
                pair_data = abstract_pair_data(data, z_emb_pair)  # 不用feature，就用z_emb替代
            h_graphormer = self.graphormer(pair_data)
            h_src_graphormer = h_graphormer[:, 0, :]
            h_dst_graphormer = h_graphormer[:, 1, :]
            h_graphormer = h_src_graphormer * h_dst_graphormer
        else:  # sum pooling
            h = global_add_pool(h, batch)
        h = torch.cat((h, h_graphormer), dim=-1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h

class GCNFFNGraphormer(torch.nn.Module):  # 必须有use_feature，输入FFN
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, use_feature_GT=True, node_embedding=None, dropout=0.5):
        super(GCNFFNGraphormer, self).__init__()
        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels, feat_channels = hidden_channels, train_dataset.num_features
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin01 = Linear(feat_channels, hidden_channels)  # FFN编码feature
        self.lin02 = Linear(2 * hidden_channels, hidden_channels)
        self.dropout = dropout
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        input_dim = train_dataset.num_features if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=hidden_channels,
                                     ffn_dim=hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.graphormer.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            h = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            h = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            h = torch.cat([h, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            h = torch.cat([h, x_rpe_emb], 1)

        for conv in self.convs[:-1]:
            h = conv(h, edge_index, edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index, edge_weight)

        ffn_feat = x.to(torch.float)
        # linear -> concat -> linear
        h_feature = F.relu(self.lin01(ffn_feat))  # FFN单独编码feature
        h = torch.cat((h, h_feature), dim=1)
        h = F.relu(self.lin02(h))

        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            h_src = h[center_indices]
            h_dst = h[center_indices + 1]
            h = h_src * h_dst
            if self.use_feature_GT:
                pair_data = abstract_pair_data(data)
            else:
                z_emb_src = z_emb[center_indices]
                z_emb_dst = z_emb[center_indices + 1]
                z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
                pair_data = abstract_pair_data(data, z_emb_pair)  # 不用feature，就用z_emb替代
            h_graphormer = self.graphormer(pair_data)
            h_src_graphormer = h_graphormer[:, 0, :]
            h_dst_graphormer = h_graphormer[:, 1, :]
            h_graphormer = h_src_graphormer * h_dst_graphormer
        else:  # sum pooling
            h = global_add_pool(h, batch)
        h = torch.cat((h, h_graphormer), dim=-1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h

class GCNGraphormer_noNeigFeat(torch.nn.Module):  # 这里的use_feature是FFN里，GCN是不要feature的
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset,  # z_emb_agg和node_embedding至少要有一个
                 use_feature=False, use_feature_GT=True, node_embedding=None, z_emb_agg=True, dropout=0.5):
        super(GCNGraphormer_noNeigFeat, self).__init__()
        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.node_embedding = node_embedding
        self.z_emb_agg = z_emb_agg
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels, self.feat_channels = 0, 0
        if self.use_feature:
            self.feat_channels += train_dataset.num_features
        if self.z_emb_agg:
            initial_channels += hidden_channels
        else:  # 如果z_emb不做GCN聚合，那就要加到后面FFN里编码
            self.feat_channels += hidden_channels
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))
        if not initial_channels:  # z_emb_agg和node_embedding至少要有一个
            print('z_emb or node_embedding for GCN!')
            import sys; sys.exit()
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        if self.feat_channels:  # feat_channels如果为0，就不要FFN了
            self.lin01 = Linear(self.feat_channels, hidden_channels)  # FFN单独编码feature(和z_emb)
            self.lin02 = Linear(2 * hidden_channels, hidden_channels)

        self.dropout = dropout
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        input_dim = train_dataset.num_features if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=hidden_channels,
                                     ffn_dim=hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.graphormer.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            if self.z_emb_agg:
                h = torch.cat([z_emb, n_emb], 1)
            else:
                h = n_emb
        else:  # 没node_embedding，就必须有z_emb_agg
            h = z_emb
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            h = torch.cat([h, x_rpe_emb], 1)

        for conv in self.convs[:-1]:
            h = conv(h, edge_index, edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index, edge_weight)

        if self.feat_channels:  # feat_channels如果为0，就不要FFN了
            ffn_feat = x.to(torch.float) if self.z_emb_agg else torch.cat((x.to(torch.float), z_emb), 1)
            # linear -> concat -> linear
            h_feature = F.relu(self.lin01(ffn_feat))  # FFN单独编码feature(和z_emb)
            h = torch.cat((h, h_feature), dim=1)
            h = F.relu(self.lin02(h))
            # # linear -> linear -> concat
            # h_feature = F.relu(self.lin01(ffn_feat))
            # h = F.relu(self.lin02(h_feature))
            # h = torch.cat((h, h_feature), dim=1)
            # concat -> linear -> linear
            # h = torch.cat((h, ffn_feat), dim=1)
            # h = F.relu(self.lin01(h))
            # h = F.relu(self.lin02(h))
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            h_src = h[center_indices]
            h_dst = h[center_indices + 1]
            h = h_src * h_dst
            # pdb.set_trace()
            if self.use_feature_GT:
                pair_data = abstract_pair_data(data)
            else:
                z_emb_src = z_emb[center_indices]
                z_emb_dst = z_emb[center_indices + 1]
                z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
                pair_data = abstract_pair_data(data, z_emb_pair)  # 不用feature，就用z_emb替代
            h_graphormer = self.graphormer(pair_data)
            h_src_graphormer = h_graphormer[:, 0, :]
            h_dst_graphormer = h_graphormer[:, 1, :]
            h_graphormer = h_src_graphormer * h_dst_graphormer
        else:  # sum pooling
            h = global_add_pool(h, batch)
        h = torch.cat((h, h_graphormer), dim=-1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h

class SingleFFN(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(SingleFFN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.ffns = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            # self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim))

        self.ffns.append(Linear(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.ffns.append(Linear(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for ffn in self.ffns:
            ffn.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            x = torch.cat([x, x_rpe_emb], 1)

        for ffn in self.ffns[:-1]:
            x = ffn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ffns[-1](x)

        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = torch.cat([x_src, x_dst], 1)
            # x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x

class FFNGraphormer(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, use_feature_GT=True, node_embedding=None, dropout=0.5):
        super(FFNGraphormer, self).__init__()
        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.ffns = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        self.ffns.append(Linear(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.ffns.append(Linear(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        input_dim = train_dataset.num_features if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=hidden_channels,
                                     ffn_dim=hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def reset_parameters(self):
        for ffn in self.ffns:
            ffn.reset_parameters()
        self.graphormer.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            h = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            h = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            h = torch.cat([h, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            h = torch.cat([h, x_rpe_emb], 1)

        for ffn in self.ffns[:-1]:
            h = ffn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.ffns[-1](h)

        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            h_src = h[center_indices]
            h_dst = h[center_indices + 1]
            h = h_src * h_dst
            if self.use_feature_GT:
                pair_data = abstract_pair_data(data)
            else:
                z_emb_src = z_emb[center_indices]
                z_emb_dst = z_emb[center_indices + 1]
                z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
                pair_data = abstract_pair_data(data, z_emb_pair)  # 不用feature，就用z_emb替代
            h_graphormer = self.graphormer(pair_data)
            h_src_graphormer = h_graphormer[:, 0, :]
            h_dst_graphormer = h_graphormer[:, 1, :]
            h_graphormer = h_src_graphormer * h_dst_graphormer
        else:  # sum pooling
            h = global_add_pool(h, batch)
        h = torch.cat((h, h_graphormer), dim=-1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h


class DGCNNGraphormer(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None,
                 GNN=GCNConv, NGNN=NGNN_GCNConv, use_feature=False, use_feature_GT=True,
                 node_embedding=None, readout_type=False):
        super(DGCNNGraphormer, self).__init__()

        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.node_embedding = node_embedding
        self.readout_type = readout_type

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                num_nodes = []
                for i, data in enumerate(iter(train_dataset)):
                    num_nodes += [data.num_nodes]
                    if i >= 1000:
                        break
                num_nodes = sorted(num_nodes)
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        if args.use_ignn:
            self.convs.append(NGNN(initial_channels, hidden_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(NGNN(hidden_channels, hidden_channels, hidden_channels))
            self.convs.append(NGNN(hidden_channels, hidden_channels, 1))
        else:
            self.convs.append(GNN(initial_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        mlp_hidden_channels = args.mlp_hidden_channels
        self.lin_align = Linear(dense_dim, mlp_hidden_channels)
        if self.readout_type == 0:
            self.lin1 = Linear(2*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 1:
            h = torch.cat((h_src, h_dst, h, h_src_graphormer, h_dst_graphormer), dim=-1)
            self.lin1 = Linear(2*total_latent_dim + 3*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 2:
            self.lin1 = Linear(3*mlp_hidden_channels, mlp_hidden_channels)

        self.lin2 = Linear(mlp_hidden_channels, 1)

        input_dim = train_dataset.num_features if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=mlp_hidden_channels,
                                     ffn_dim=mlp_hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def forward(self, data):
        # pdb.set_trace()
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            h = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            h = z_emb
        # [2747, 32]
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            h = torch.cat([h, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            h = torch.cat([h, x_rpe_emb], 1)
        hs = [h]
        for conv in self.convs:
            hs += [torch.tanh(conv(hs[-1], edge_index, edge_weight))]
        h = torch.cat(hs[1:], dim=-1)  # h: [num_nodes, 3*input_dim+1] [2747, 97]

        _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
        if self.readout_type:
            h_src = h[center_indices]
            h_dst = h[center_indices + 1]

        # Global pooling.
        h = global_sort_pool(h, batch, self.k)  # h: [num_graphs, k * hidden] [4, 11155]
        h = h.unsqueeze(1)  # [num_graphs, 1, k * hidden] [4, 1, 11155]
        h = F.relu(self.conv1(h))
        h = self.maxpool1d(h)
        # [4, 16, 57]
        h = F.relu(self.conv2(h))
        # [4, 32, 53]
        h = h.view(h.size(0), -1)  # [num_graphs, dense_dim]
        # [4, 1696]
        h = F.relu(self.lin_align(h))
        # [4, 128]

        # Graphormer embedding
        if self.use_feature_GT:
            pair_data = abstract_pair_data(data)
        else:
            z_emb_src = z_emb[center_indices]
            z_emb_dst = z_emb[center_indices + 1]
            z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
            pair_data = abstract_pair_data(data, z_emb_pair)  # 不用feature，就用z_emb替代
        if pair_data.x.dtype == torch.long:
            pair_data.x = pair_data.x.float()
        h_graphormer = self.graphormer(pair_data)
        h_src_graphormer = h_graphormer[:,0,:]
        h_dst_graphormer = h_graphormer[:,1,:]

        # Aggregation
        if self.readout_type == 0:
            h = torch.cat((h, h_src_graphormer*h_dst_graphormer), dim=-1)
        elif self.readout_type == 1:
            h = torch.cat((h_src, h_dst, h, h_src_graphormer, h_dst_graphormer), dim=-1)
        elif self.readout_type == 2:
            h = torch.cat((h, h_src_graphormer, h_dst_graphormer), dim=-1)

        # h: [4, 160]; lin1: Linear(in_features=256, out_features=128, bias=True)
        # MLP.
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h

class DGCNNGraphormer_noNeigFeat(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None,
                 GNN=GCNConv, NGNN=NGNN_GCNConv, use_feature=False, use_feature_GT=True,
                 node_embedding=None, readout_type=False):
        super(DGCNNGraphormer_noNeigFeat, self).__init__()

        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.node_embedding = node_embedding
        self.readout_type = readout_type

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                num_nodes = []
                for i, data in enumerate(iter(train_dataset)):
                    num_nodes += [data.num_nodes]
                    if i >= 1000:
                        break
                num_nodes = sorted(num_nodes)
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels = hidden_channels
        assert train_dataset.num_features > 0
        feat_channels = train_dataset.num_features
        self.lin01 = Linear(feat_channels, hidden_channels)  # FFN单独编码feature
        self.lin02 = Linear(num_layers * hidden_channels + 1 + hidden_channels, num_layers * hidden_channels + 1)  # 3*32+1+32, 3*32+1


        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        if args.use_ignn:
            self.convs.append(NGNN(initial_channels, hidden_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(NGNN(hidden_channels, hidden_channels, hidden_channels))
            self.convs.append(NGNN(hidden_channels, hidden_channels, 1))
        else:
            self.convs.append(GNN(initial_channels, hidden_channels))
            for i in range(0, num_layers-1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        mlp_hidden_channels = args.mlp_hidden_channels
        self.lin_align = Linear(dense_dim, mlp_hidden_channels)
        if self.readout_type == 0:
            self.lin1 = Linear(2*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 1:
            h = torch.cat((h_src, h_dst, h, h_src_graphormer, h_dst_graphormer), dim=-1)
            self.lin1 = Linear(2*total_latent_dim + 3*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 2:
            self.lin1 = Linear(3*mlp_hidden_channels, mlp_hidden_channels)

        self.lin2 = Linear(mlp_hidden_channels, 1)

        input_dim = train_dataset.num_features if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=mlp_hidden_channels,
                                     ffn_dim=mlp_hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def forward(self, data):
        # pdb.set_trace()
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        h = z_emb
        ffn_feat = x.to(torch.float)

        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            h = torch.cat([h, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            h = torch.cat([h, x_rpe_emb], 1)
        hs = [h]
        for conv in self.convs:
            hs += [torch.tanh(conv(hs[-1], edge_index, edge_weight))]
        h = torch.cat(hs[1:], dim=-1)  # h: [num_nodes, 3*input_dim+1] [2747, 97]

        # linear -> concat -> linear
        h_feature = F.relu(self.lin01(ffn_feat))  # FFN单独编码feature
        h = torch.cat((h, h_feature), dim=1)
        h = F.relu(self.lin02(h))

        _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
        if self.readout_type:
            h_src = h[center_indices]
            h_dst = h[center_indices + 1]

        # Global pooling.
        h = global_sort_pool(h, batch, self.k)  # h: [num_graphs, k * hidden] [4, 11155]
        h = h.unsqueeze(1)  # [num_graphs, 1, k * hidden] [4, 1, 11155]
        h = F.relu(self.conv1(h))
        h = self.maxpool1d(h)
        # [4, 16, 57]
        h = F.relu(self.conv2(h))
        # [4, 32, 53]
        h = h.view(h.size(0), -1)  # [num_graphs, dense_dim]
        # [4, 1696]
        h = F.relu(self.lin_align(h))
        # [4, 128]

        # Graphormer embedding
        if self.use_feature_GT:
            pair_data = abstract_pair_data(data)
        else:
            z_emb_src = z_emb[center_indices]
            z_emb_dst = z_emb[center_indices + 1]
            z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
            pair_data = abstract_pair_data(data, z_emb_pair)  # 不用feature，就用z_emb替代
        if pair_data.x.dtype == torch.long:
            pair_data.x = pair_data.x.float()
        h_graphormer = self.graphormer(pair_data)
        h_src_graphormer = h_graphormer[:,0,:]
        h_dst_graphormer = h_graphormer[:,1,:]

        # Aggregation
        if self.readout_type == 0:
            h = torch.cat((h, h_src_graphormer*h_dst_graphormer), dim=-1)
        elif self.readout_type == 1:
            h = torch.cat((h_src, h_dst, h, h_src_graphormer, h_dst_graphormer), dim=-1)
        elif self.readout_type == 2:
            h = torch.cat((h, h_src_graphormer, h_dst_graphormer), dim=-1)

        # h: [4, 160]; lin1: Linear(in_features=256, out_features=128, bias=True)
        # MLP.
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h


class NGNNDGCNNGraphormer(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, k=62, feature_dim=128,
                 use_feature=True, use_feature_GT=True,
                 node_embedding=None, readout_type=False):
        super(NGNNDGCNNGraphormer, self).__init__()
        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.readout_type = readout_type

        self.ngnndgcnn = ngnn_models.DGCNN(
                args.hidden_channels,
                args.num_layers,
                args.max_z,
                k,
                feature_dim=feature_dim,
                dropout=args.dropout,
                ngnn_type=args.ngnn_type,
                num_ngnn_layers=args.num_ngnn_layers,
                output_embedding=True
            )

        mlp_hidden_channels = args.mlp_hidden_channels
        dense_dim = self.ngnndgcnn.dense_dim
        self.lin_align = Linear(dense_dim, mlp_hidden_channels)
        if self.readout_type == 0:
            self.lin1 = Linear(2*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 1:
            h = torch.cat((h_src, h_dst, h, h_src_graphormer, h_dst_graphormer), dim=-1)
            self.lin1 = Linear(2*total_latent_dim + 3*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 2:
            self.lin1 = Linear(3*mlp_hidden_channels, mlp_hidden_channels)
        self.lin2 = Linear(mlp_hidden_channels, 1)

        input_dim = feature_dim if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=mlp_hidden_channels,
                                     ffn_dim=mlp_hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def forward(self, g, z, x=None, edge_weight=None):
        # ngnndgcnn embedding
        h = self.ngnndgcnn(g, z, x, edge_weight=edge_weight)  # [num_graphs, dense_dim]
        # [4, 1696]
        h = F.relu(self.lin_align(h))
        # [4, 128]

        # import pdb; pdb.set_trace()
        # Graphormer embedding
        # if self.use_feature_GT:
        #     pair_data = abstract_pair_data(g)
        # else:
        #     z_emb_src = z_emb[center_indices]
        #     z_emb_dst = z_emb[center_indices + 1]
        #     z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
        #     pair_data = abstract_pair_data(g, z_emb_pair)  # 不用feature，就用z_emb替代
        if g.x.dtype == torch.long:
            g.x = g.x.float()
        h_graphormer = self.graphormer(g)
        h_src_graphormer = h_graphormer[:,0,:]
        h_dst_graphormer = h_graphormer[:,1,:]

        # Aggregation
        if self.readout_type == 0:
            h = torch.cat((h, h_src_graphormer * h_dst_graphormer), dim=-1)
        elif self.readout_type == 1:
            h = torch.cat((h_src, h_dst, h, h_src_graphormer, h_dst_graphormer), dim=-1)
        elif self.readout_type == 2:
            h = torch.cat((h, h_src_graphormer, h_dst_graphormer), dim=-1)

        # MLP.
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h

class NGNNDGCNNGraphormer_noNeigFeat(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, k=62, feature_dim=128,
                 use_feature=True, use_feature_GT=True,
                 node_embedding=None, readout_type=False):
        super(NGNNDGCNNGraphormer_noNeigFeat, self).__init__()
        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.readout_type = readout_type

        self.ngnndgcnn_noneigfeat = ngnn_models.DGCNN_noNeigFeat(
                args.hidden_channels,
                args.num_layers,
                args.max_z,
                k,
                feature_dim=feature_dim,
                dropout=args.dropout,
                ngnn_type=args.ngnn_type,
                num_ngnn_layers=args.num_ngnn_layers,
                output_embedding=True
            )

        mlp_hidden_channels = args.mlp_hidden_channels
        dense_dim = self.ngnndgcnn_noneigfeat.dense_dim
        self.lin_align = Linear(dense_dim, mlp_hidden_channels)
        if self.readout_type == 0:
            self.lin1 = Linear(2*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 1:
            h = torch.cat((h_src, h_dst, h, h_src_graphormer, h_dst_graphormer), dim=-1)
            self.lin1 = Linear(2*total_latent_dim + 3*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 2:
            self.lin1 = Linear(3*mlp_hidden_channels, mlp_hidden_channels)
        self.lin2 = Linear(mlp_hidden_channels, 1)

        input_dim = feature_dim if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=mlp_hidden_channels,
                                     ffn_dim=mlp_hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def forward(self, g, z, x=None, edge_weight=None):
        # ngnndgcnn_noneigfeat embedding
        h = self.ngnndgcnn_noneigfeat(g, z, x, edge_weight=edge_weight)  # [num_graphs, dense_dim]
        # [64, 896]
        h = F.relu(self.lin_align(h))
        # [64, 128]

        # import pdb; pdb.set_trace()
        # Graphormer embedding
        # if self.use_feature_GT:
        #     pair_data = abstract_pair_data(g)
        # else:
        #     z_emb_src = z_emb[center_indices]
        #     z_emb_dst = z_emb[center_indices + 1]
        #     z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
        #     pair_data = abstract_pair_data(g, z_emb_pair)  # 不用feature，就用z_emb替代
        if g.x.dtype == torch.long:
            g.x = g.x.float()
        h_graphormer = self.graphormer(g)
        h_src_graphormer = h_graphormer[:,0,:]
        h_dst_graphormer = h_graphormer[:,1,:]

        # Aggregation
        if self.readout_type == 0:
            h = torch.cat((h, h_src_graphormer * h_dst_graphormer), dim=-1)
        elif self.readout_type == 1:
            h = torch.cat((h_src, h_dst, h, h_src_graphormer, h_dst_graphormer), dim=-1)
        elif self.readout_type == 2:
            h = torch.cat((h, h_src_graphormer, h_dst_graphormer), dim=-1)

        # MLP.
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h

class SingleGraphormer(torch.nn.Module):  # use_feature_GT没用，是use_feature来控制
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset=None,
                 use_feature=False, use_feature_GT=True, node_embedding=None, dropout=0.5):
        super(SingleGraphormer, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        input_dim = train_dataset.num_features
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=hidden_channels,
                                     ffn_dim=hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def forward(self, data):
        # x = data.x
        # z = data.z
        # edge_index = data.edge_index
        batch = data.batch
        # edge_weight = data.edge_weight
        # node_id = data.node_id

        # z_emb = self.z_embedding(z)
        # if z_emb.ndim == 3:  # in case z has multiple integer labels
        #     z_emb = z_emb.sum(dim=1)
        # if self.use_feature and x is not None:
        #     h = torch.cat([z_emb, x.to(torch.float)], 1)
        # else:
        #     h = z_emb
        # if self.node_embedding is not None and node_id is not None:
        #     n_emb = self.node_embedding(node_id)
        #     h = torch.cat([h, n_emb], 1)
        # if self.use_rpe:
        #     x_rpe = data.x_rpe
        #     x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
        #     h = torch.cat([h, x_rpe_emb], 1)

        pair_data = abstract_pair_data(data)
        _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
        h_graphormer = self.graphormer(pair_data)
        h_src_graphormer = h_graphormer[:, 0, :]
        h_dst_graphormer = h_graphormer[:, 1, :]
        h = h_src_graphormer * h_dst_graphormer
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h
