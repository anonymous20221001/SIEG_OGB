import math
import numpy as np
import torch
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU, 
                      Sequential, BatchNorm1d as BN)
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, SAGEConv, GINConv, 
                                global_sort_pool, global_add_pool, global_mean_pool)
from torch_geometric.data import Data
from graphormer import Graphormer
import pdb


# x: (N, 128)，点特征
# z: (N)，DRNL，每个点一维
# z_embedding: (1000, 32)，z的lookup table
# edge_index: (2, Ne)
# batch: (N)，代表每个节点属于哪一个子图的标签，类似[0, .., 0, 1, ..., 1, 2, ..., 31]
# node_embedding: 对全局的node_id进行编码得到的embedding
# node_id: 全局点序号

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset, 
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(GCN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
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
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset=None, 
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(SAGE, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
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
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None, 
                 GNN=GCNConv, use_feature=False, 
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
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

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


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, 
                 jk=True, train_eps=False):
        super(GIN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
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


def abstract_pair_data(data):
    pair_data = Data(x=data.pair_x, edge_index=data.pair_edge_idx)
    pair_data.attn_bias = data.pair_attn_bias
    pair_data.in_degree = data.pair_in_degree
    pair_data.out_degree = data.pair_out_degree
    pair_data.len_shortest_path = data.pair_len_shortest_path
    pair_data.num_shortest_path = data.pair_num_shortest_path
    pair_data.undir_jac = data.pair_undir_jac
    pair_data.undir_aa = data.pair_undir_aa

    return pair_data

class GCNGraphormer(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, train_dataset, 
                 use_feature=False, node_embedding=None, dropout=0.5):
        super(GCNGraphormer, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        n_layers = 3
        num_node_feat = train_dataset.num_features
        num_heads = 32
        hidden_dim = hidden_channels
        ffn_dim = hidden_channels
        use_num_spd = args.use_num_spd
        use_cnb_jac = args.use_cnb_jac
        use_cnb_aa = args.use_cnb_aa
        use_degree = args.use_degree
        self.graphormer = Graphormer(n_layers, num_node_feat, num_heads, hidden_dim, ffn_dim, use_num_spd, use_cnb_jac, use_cnb_aa, use_degree)

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
            pair_data = abstract_pair_data(data)
            h_graphormer = self.graphormer(pair_data)
            h_src_graphormer = h_graphormer[:,0,:]
            h_dst_graphormer = h_graphormer[:,1,:]
        else:  # sum pooling
            h = global_add_pool(h, batch)
        h = torch.cat((h, h_src_graphormer*h_dst_graphormer), dim=-1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h


class DGCNNGraphormer(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None, 
                 GNN=GCNConv, use_feature=False, 
                 node_embedding=None, readout_type=False):
        super(DGCNNGraphormer, self).__init__()

        self.use_feature = use_feature
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

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

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
        mlp_hidden_channels = 128
        self.lin_align = Linear(dense_dim, mlp_hidden_channels)
        if self.readout_type == 0:
            self.lin1 = Linear(2*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 1:
            h = torch.cat((h_src, h_dst, h, h_src_graphormer, h_dst_graphormer), dim=-1)
            self.lin1 = Linear(2*total_latent_dim + 3*mlp_hidden_channels, mlp_hidden_channels)
        elif self.readout_type == 2:
            self.lin1 = Linear(3*mlp_hidden_channels, mlp_hidden_channels)

        self.lin2 = Linear(mlp_hidden_channels, 1)

        n_layers = 3

        num_node_feat = train_dataset.num_features
        num_heads = 32
        hidden_dim = mlp_hidden_channels
        ffn_dim = mlp_hidden_channels
        use_num_spd = args.use_num_spd
        use_cnb_jac = args.use_cnb_jac
        use_cnb_aa = args.use_cnb_aa
        use_degree = args.use_degree
        self.graphormer = Graphormer(n_layers, num_node_feat, num_heads, hidden_dim, ffn_dim, use_num_spd, use_cnb_jac, use_cnb_aa, use_degree)

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
        hs = [h]

        for conv in self.convs:
            hs += [torch.tanh(conv(hs[-1], edge_index, edge_weight))]
        h = torch.cat(hs[1:], dim=-1)

        if self.readout_type:
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            h_src = h[center_indices]
            h_dst = h[center_indices + 1]

        # Global pooling.
        h = global_sort_pool(h, batch, self.k)
        h = h.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        h = F.relu(self.conv1(h))
        h = self.maxpool1d(h)
        h = F.relu(self.conv2(h))
        h = h.view(h.size(0), -1)  # [num_graphs, dense_dim]
        h = F.relu(self.lin_align(h))

        # Graphormer embedding
        pair_data = abstract_pair_data(data)
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

        # MLP.
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h


