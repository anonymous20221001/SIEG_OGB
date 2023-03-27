import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from timer_guard import TimerGuard
import pdb

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, mul_bias=False, grpe_cross=False):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)
        self.mul_bias = mul_bias
        self.grpe_cross = grpe_cross

    def reset_parameters(self):
        self.linear_q.reset_parameters()
        self.linear_k.reset_parameters()
        self.linear_v.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, q, k, v, attn_bias=None, spatial_pos_query=None, spatial_pos_key=None):
        orig_q_size = q.size()
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.att_size)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.att_size)

        # q, k, v: [n_graph, num_heads, n_node+1, att_size]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)

        if self.grpe_cross:
            spatial_bias_query = torch.matmul(q.unsqueeze(3), spatial_pos_query.transpose(3, 4)).squeeze()
            spatial_bias_key = torch.matmul(k.unsqueeze(3), spatial_pos_key.transpose(3, 4)).squeeze()
            spatial_bias = spatial_bias_query + spatial_bias_key
            a = torch.matmul(q, k.transpose(2, 3)) + spatial_bias  # 不加edge_bais
            a = a * self.scale
        else:
            # Scaled Dot-Product Attention.
            # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
            q = q * self.scale
            a = torch.matmul(q, k.transpose(2, 3))  # [n_graph, num_heads, n_node+1, n_node+1]

        if attn_bias is not None:
            if not self.mul_bias:
                a += attn_bias
            else:
                a *= attn_bias

        a = torch.softmax(a, dim=3)
        a = self.att_dropout(a)
        # a: [n_graph, num_heads, n_node+1, n_node+1]
        # v: [n_graph, num_heads, n_node+1, att_size]
        # x: [n_graph, num_heads, n_node+1, att_size]
        x = a.matmul(v)
        # if self.grpe_cross:
        #     x += a.unsqueeze(3).matmul(spatial_pos_query).squeeze()
        x = x.transpose(1, 2).contiguous()  # [n_graph, n_node+1, num_heads, att_size]
        x = x.view(batch_size, -1, self.num_heads * self.att_size)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, mul_bias=False, grpe_cross=False):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads, mul_bias, grpe_cross)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def reset_parameters(self):
        self.self_attention_norm.reset_parameters()
        self.self_attention.reset_parameters()

        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, attn_bias=None, spatial_pos_query=None, spatial_pos_key=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, spatial_pos_query, spatial_pos_key)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Graphormer(nn.Module):
    def __init__(
        self,
        n_layers,
        input_dim,
        num_heads,
        hidden_dim,
        ffn_dim,
        grpe_cross=False,
        use_len_spd=True,
        use_num_spd=False,
        use_cnb_jac=False,
        use_cnb_aa=False,
        use_cnb_ra=False,
        use_degree=False,
        mul_bias=False,
        gravity_type=0,
        dropout_rate=0.1,
        intput_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        multi_hop_max_dist=20,
    ):
        super(Graphormer, self).__init__()
        self.grpe_cross = grpe_cross
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.hidden_dim = hidden_dim
        self.att_size = hidden_dim // num_heads
        self.use_len_spd = use_len_spd
        self.use_num_spd = use_num_spd
        self.use_cnb_jac = use_cnb_jac
        self.use_cnb_aa = use_cnb_aa
        self.use_cnb_ra = use_cnb_ra
        self.use_degree = use_degree
        self.gravity_type = gravity_type
        self.atom_encoder = nn.Linear(input_dim, hidden_dim)
        # self.edge_encoder = nn.Embedding(64, num_heads, padding_idx=0)
        # self.edge_type = edge_type
        # if self.edge_type == 'multi_hop':
        #    self.edge_dis_encoder = nn.Embedding(
        #        40 * num_heads * num_heads, 1)
        if self.grpe_cross:  # 按grpe的特征交叉方式，需要query和key两套结构特征embedding
            if use_len_spd:
                self.len_shortest_path_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.len_shortest_path_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_num_spd:
                self.num_shortest_path_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.num_shortest_path_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_cnb_jac:
                self.undir_jac_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.undir_jac_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_cnb_aa:
                self.undir_aa_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.undir_aa_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_cnb_ra:
                self.undir_ra_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.undir_ra_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            # 固定0，不可学习
            self.padding1 = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1, self.att_size), requires_grad=False)
            self.padding2 = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1, self.att_size), requires_grad=False)
            # 随机初始化，可学习
            # self.padding1 = nn.Parameter(torch.randn(1, num_heads, 1, 1, self.att_size))
            # self.padding2 = nn.Parameter(torch.randn(1, num_heads, 1, 1, self.att_size))
        else:
            if use_len_spd:
                self.len_shortest_path_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_num_spd:
                self.num_shortest_path_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_cnb_jac:
                self.undir_jac_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_cnb_aa:
                self.undir_aa_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_cnb_ra:
                self.undir_ra_encoder = nn.Embedding(40, num_heads, padding_idx=0)
        if use_degree:  # 点上的特征不需要两套
            self.in_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.undir_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)

        num_edge_types = 1
        max_len_rule = 3
        num_rules = pow(num_edge_types*2, max_len_rule+1) - 2
        depth_rules = []
        for depth_rule in range(1, max_len_rule+1):
            depth_rules += [depth_rule] * pow(num_edge_types*2, depth_rule)
        self.depth_rules = torch.Tensor(depth_rules).long()
        self.gravity_scale = 1. / torch.pow(self.depth_rules, 2)
        if gravity_type in [1, 2, 3]:
            self.path_rule_lin = nn.Linear(num_rules, num_heads)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, mul_bias, grpe_cross)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)


    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.final_ln.reset_parameters()
        self.atom_encoder.reset_parameters()
        # self.edge_encoder.reset_parameters()
        # self.edge_type = edge_type
        # if self.edge_type == 'multi_hop':
        #    self.edge_dis_encoder.reset_parameters()
        if self.grpe_cross:
            if self.use_len_spd:
                self.len_shortest_path_encoder_query.reset_parameters()
                self.len_shortest_path_encoder_key.reset_parameters()
            if self.use_num_spd:
                self.num_shortest_path_encoder_query.reset_parameters()
                self.num_shortest_path_encoder_key.reset_parameters()
            if self.use_cnb_jac:
                self.undir_jac_encoder_query.reset_parameters()
                self.undir_jac_encoder_key.reset_parameters()
            if self.use_cnb_aa:
                self.undir_aa_encoder_query.reset_parameters()
                self.undir_aa_encoder_key.reset_parameters()
            if self.use_cnb_ra:
                self.undir_ra_encoder_query.reset_parameters()
                self.undir_ra_encoder_key.reset_parameters()
        else:
            if self.use_len_spd:
                self.len_shortest_path_encoder.reset_parameters()
            if self.use_num_spd:
                self.num_shortest_path_encoder.reset_parameters()
            if self.use_cnb_jac:
                self.undir_jac_encoder.reset_parameters()
            if self.use_cnb_aa:
                self.undir_aa_encoder.reset_parameters()
            if self.use_cnb_ra:
                self.undir_ra_encoder.reset_parameters()
        if self.gravity_type in [1, 2, 3]:
            self.path_rule_lin.reset_parameters()
        if self.use_degree:
            self.in_degree_encoder.reset_parameters()
            self.out_degree_encoder.reset_parameters()
            self.undir_degree_encoder.reset_parameters()

    @TimerGuard('forward', 'utils')
    def forward(self, data, perturb=None):
        # attn_bias：图中节点对之间的最短路径距离超过最短路径限制最大距离(len_shortest_path_max)的位置为-∞，其余位置为0，形状为(n_graph, n_node+1, n_node+1)
        # len_shortest_path：图中节点对之间的最短路径长度，形状为(n_graph, n_node, n_node)
        # x：图中节点的特征，形状为(n_graph, n_node, n_node_features)
        # in_degree：图中节点的入度，形状为(n_graph, n_node)
        # out_degree：图中节点的出度，形状为(n_graph, n_node)
        # edge_input：图中节点对之间的最短路径(限制最短路径最大跳数为multi_hop_max_dist)上的边的特征，形状为(n_graph, n_node, n_node, multi_hop_max_dist, n_edge_features)
        # attn_edge_type：图的边特征，形状为(n_graph, n_node, n_node, n_edge_features)
        device = data.attn_bias.device
        x = data.x  # feature: [n_graph, n_head(2), num_feat]; z_emb: [n_graph, n_head(2), dim_hidden]
        attn_bias = data.attn_bias
        # edge_input = data.edge_input
        # graph_attn_bias
        # 添加虚拟节点表示全图特征表示，之后按照图中正常节点处理
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # 空间编码,节点之间最短路径长度对应的可学习标量
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = torch.zeros([n_graph, self.num_heads, n_node, n_node], device=device)
        spatial_pos_query = torch.zeros([n_graph, self.num_heads, n_node, n_node, self.att_size], device=device)
        spatial_pos_key = torch.zeros([n_graph, self.num_heads, n_node, n_node, self.att_size], device=device)

        if self.grpe_cross:
            # [n_graph, n_node, n_node] -> [n_graph, n_head, n_node, n_node, att_size]
            if self.use_len_spd:
                len_shortest_path = torch.clamp(data.len_shortest_path, min=0, max=39).long()
                spatial_pos_query = self.len_shortest_path_encoder_query(len_shortest_path).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key = self.len_shortest_path_encoder_key(len_shortest_path).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            if self.use_num_spd:
                num_shortest_path = torch.clamp(data.num_shortest_path, min=0, max=39).long()
                spatial_pos_query += self.num_shortest_path_encoder_query(num_shortest_path.long()).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key += self.num_shortest_path_encoder_key(num_shortest_path.long()).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            if self.use_cnb_jac:
                undir_jac_enc = torch.clamp(data.undir_jac*30, min=0, max=39).long()
                spatial_pos_query += self.undir_jac_encoder_query(undir_jac_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key += self.undir_jac_encoder_key(undir_jac_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            if self.use_cnb_aa:
                undir_aa_enc = torch.clamp(data.undir_aa*10, min=0, max=39).long()
                spatial_pos_query += self.undir_aa_encoder_query(undir_aa_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key += self.undir_aa_encoder_key(undir_aa_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            if self.use_cnb_ra:
                undir_ra_enc = torch.clamp(data.undir_ra*10, min=0, max=39).long()
                spatial_pos_query += self.undir_ra_encoder_query(undir_ra_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key += self.undir_ra_encoder_key(undir_ra_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            padding1_batch, padding2_batch = self.padding1.repeat(n_graph, 1, 1, n_node, 1), self.padding2.repeat(n_graph, 1, n_node+1, 1, 1)
            # [n_graph, n_head, n_node, n_node, att_size] -> [n_graph, n_head, n_node+1, n_node+1, att_size]
            spatial_pos_query = torch.cat((padding2_batch, torch.cat((padding1_batch, spatial_pos_query), dim=2)), dim=3)
            spatial_pos_key = torch.cat((padding2_batch, torch.cat((padding1_batch, spatial_pos_key), dim=2)), dim=3)
        else:
            # spatial pos：空间编码,节点之间最短路径长度对应的可学习标量
            # [n_graph, n_node, n_node] -> [n_graph, n_head, n_node, n_node]
            # import pdb; pdb.set_trace()
            if self.use_len_spd:
                len_shortest_path = torch.clamp(data.len_shortest_path, min=0, max=39).long()
                spatial_pos_bias = self.len_shortest_path_encoder(len_shortest_path).permute(0, 3, 1, 2)
            if self.use_num_spd:
                num_shortest_path = torch.clamp(data.num_shortest_path, min=0, max=39).long()
                spatial_pos_bias += self.num_shortest_path_encoder(num_shortest_path.long()).permute(0, 3, 1, 2)
            if self.use_cnb_jac:
                undir_jac_enc = torch.clamp(data.undir_jac*30, min=0, max=39).long()
                spatial_pos_bias += self.undir_jac_encoder(undir_jac_enc).permute(0, 3, 1, 2)
            if self.use_cnb_aa:
                undir_aa_enc = torch.clamp(data.undir_aa*10, min=0, max=39).long()
                spatial_pos_bias += self.undir_aa_encoder(undir_aa_enc).permute(0, 3, 1, 2)
            if self.use_cnb_ra:
                undir_ra_enc = torch.clamp(data.undir_ra*10, min=0, max=39).long()
                spatial_pos_bias += self.undir_ra_encoder(undir_ra_enc).permute(0, 3, 1, 2)

        if self.gravity_type == 1:
            spatial_pos_bias = spatial_pos_bias + self.path_rule_lin(data.paths_weight).permute(0, 3, 1, 2)
        elif self.gravity_type == 2:
            gravity_scale = self.gravity_scale.to(device)
            spatial_pos_bias = spatial_pos_bias + self.path_rule_lin(data.paths_weight * gravity_scale).permute(0, 3, 1, 2)
        elif self.gravity_type == 3:
            gravity_scale = self.gravity_scale.to(device)
            spatial_pos_bias = spatial_pos_bias + self.path_rule_lin(data.paths_log_weight * gravity_scale).permute(0, 3, 1, 2)
        if not self.grpe_cross:
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        # 所有节点都和虚拟节点直接有边相连，则所有节点和虚拟节点之间的最短路径长度为1
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        x = x.to(torch.float32)
        node_feature = self.atom_encoder(x)           # [n_graph, n_node, n_hidden]

        # 根据节点的入度、出度为每个节点分配两个实值嵌入向量，添加到节点特征中作为输入
        if self.use_degree:
            if hasattr(data, 'in_degree'):
                in_degree = torch.clamp(data.in_degree, min=0, max=63).long()
                out_degree = torch.clamp(data.out_degree, min=0, max=63).long()
                node_feature = node_feature + \
                    self.in_degree_encoder(in_degree) + \
                    self.out_degree_encoder(out_degree)
            else:
                undir_degree = torch.clamp(data.undir_degree, min=0, max=63).long()
                node_feature = node_feature + self.undir_degree_encoder(undir_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            if self.grpe_cross:
                output = enc_layer(output, graph_attn_bias, spatial_pos_query, spatial_pos_key)
            else:
                output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        return output
