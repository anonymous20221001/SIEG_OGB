import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from graphormer import Graphormer
import ngnndgcnn


def abstract_pair_data(data, z_emb_pair=None):
    if z_emb_pair is None:
        pair_data = Data(x=data.pair_x, z=data.pair_z, edge_index=data.pair_edge_idx)
    else:  # 传入z_emb，就用z_emb替代feature
        pair_data = Data(x=z_emb_pair, z=data.pair_z, edge_index=data.pair_edge_idx)
    for key in data.keys:
        if key.startswith('pair_') and key not in ['pair_x', 'pair_z', 'pair_edge_idx']:
            pair_data[key[5:]] = data[key]
    return pair_data

class NGNNDGCNNGraphormer(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, k=62, feature_dim=128,
                 use_feature=True, use_feature_GT=True,
                 node_embedding=None, readout_type=False):
        super(NGNNDGCNNGraphormer, self).__init__()
        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.readout_type = readout_type

        self.ngnndgcnn = ngnndgcnn.DGCNN(
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

        mlp_hidden_channels = 128
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
        h = self.ngnndgcnn(g, z, x, edge_weight=edge_weight)  # [num_graphs, dense_dim]
        h = F.relu(self.lin_align(h))
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
