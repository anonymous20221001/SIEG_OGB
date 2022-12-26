import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)
from utils import *
from models import *
import networkx as nx
from timer_guard import TimerGuard


@TimerGuard('preprocess', 'utils')
def preprocess(data, degree, **kwargs):

    use_len_spd = kwargs["use_len_spd"] if "use_len_spd" in kwargs else False
    use_num_spd = kwargs["use_num_spd"] if "use_num_spd" in kwargs else False
    use_cnb_jac = kwargs["use_cnb_jac"] if "use_cnb_jac" in kwargs else False
    use_cnb_aa = kwargs["use_cnb_aa"] if "use_cnb_aa" in kwargs else False
    use_cnb_ra = kwargs["use_cnb_ra"] if "use_cnb_ra" in kwargs else False
    use_degree = kwargs["use_degree"] if "use_degree" in kwargs else False
    gravity_type = kwargs["gravity_type"] if "gravity_type" in kwargs else 0
    directed = kwargs["directed"] if "directed" in kwargs else False

    device = data.edge_index.device
    center_indices = np.array([0])
    num_pair = len(center_indices)

    # estimate shortest path using undirected graph
    if gravity_type == 0:
        G_undirected = to_networkx(Data(torch.zeros_like(data.node_id), torch.cat([data.edge_index.cpu(), data.edge_index.cpu()[[1, 0], :]], dim=-1)), to_undirected=True)

    pair_len_shortest_path = torch.zeros((num_pair, 2, 2), dtype=torch.long)
    pair_num_shortest_path = torch.zeros((num_pair, 2, 2), dtype=torch.long)
    pair_undir_jac = torch.zeros((num_pair, 2, 2), dtype=torch.float32)
    pair_undir_aa = torch.zeros((num_pair, 2, 2), dtype=torch.float32)
    pair_undir_ra = torch.zeros((num_pair, 2, 2), dtype=torch.float32)

    for i in range(num_pair):
        src_idx = center_indices[i]
        dst_idx = center_indices[i] + 1
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

    n_graph = len(center_indices)
    n_node = 2
    data.pair_attn_bias = torch.zeros(n_graph, n_node+1, n_node+1).to(device)
    data.pair_edge_idx = torch.Tensor([[0, 1], [1, 0]]).long().unsqueeze(axis=0).expand(n_graph, -1, -1).to(device)
    x_src = data.x[center_indices]
    x_dst = data.x[center_indices+1]
    data.pair_x = torch.stack((x_src, x_dst), dim=1)
    z_src = data.z[center_indices]
    z_dst = data.z[center_indices+1]
    data.pair_z = torch.stack((z_src, z_dst), dim=1)

    return data
