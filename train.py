import argparse
import time
import os, sys
import shutil
from tqdm import tqdm
from functools import partial
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import BCEWithLogitsLoss
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from torch_geometric.datasets import Planetoid
from data_pairwise import preprocess
from utils import *
from models import *
from timer_guard import TimerGuard

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', 
                    stream=sys.stdout,
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

import argparse
import os
import sys
import time

import dgl
import torch
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset
from tqdm import tqdm

import data_prepare

class SEALOGBLDataset(Dataset):
    def __init__(
        self,
        data_pyg,
        preprocess_fn,
        root,
        graph,
        split_edge,
        percent=100,
        split="train",
        ratio_per_hop=1.0,
        directed=False,
        dynamic=True,
    ) -> None:
        super().__init__()
        self.data_pyg = data_pyg
        self.preprocess_fn = preprocess_fn
        self.root = root
        self.graph = graph
        self.split = split
        self.split_edge = split_edge
        self.percent = percent
        self.ratio_per_hop = ratio_per_hop
        self.directed = directed
        self.dynamic = dynamic

        if "weights" in self.graph.edata:
            self.edge_weights = self.graph.edata["weights"]
        else:
            self.edge_weights = None
        if "feat" in self.graph.ndata:
            self.node_features = self.graph.ndata["feat"]
        else:
            self.node_features = None

        pos_edge, neg_edge = data_prepare.get_pos_neg_edges(
            self.split, self.split_edge, self.graph, self.percent
        )
        self.links = torch.cat([pos_edge, neg_edge], 0)  # [Np + Nn, 2] [1215518, 2]
        self.labels = np.array([1] * len(pos_edge) + [0] * len(neg_edge))  # [1215518]

        if not self.dynamic:
            self.g_list, tensor_dict = self.load_cached()
            self.labels = tensor_dict["y"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.dynamic:
            g, y = self.g_list[idx], self.labels[idx]
            x = None if "x" not in g.ndata else g.ndata["x"]
            w = None if "w" not in g.edata else g.eata["w"]
            return g, g.ndata["z"], x, w, y

        src, dst = self.links[idx][0].item(), self.links[idx][1].item()
        y = self.labels[idx]  # 1
        subg = data_prepare.k_hop_subgraph(
            src, dst, 1, self.graph, self.ratio_per_hop, self.directed
        )

        # Remove the link between src and dst.
        direct_links = [[], []]
        for s, t in [(0, 1), (1, 0)]:
            if subg.has_edges_between(s, t):
                direct_links[0].append(s)
                direct_links[1].append(t)
        if len(direct_links[0]):
            subg.remove_edges(subg.edge_ids(*direct_links))

        NIDs, EIDs = subg.ndata[dgl.NID], subg.edata[dgl.EID]  # [32] [72]

        z = data_prepare.drnl_node_labeling(subg.adj(scipy_fmt="csr"), 0, 1)  # [32]
        edge_weights = (
            self.edge_weights[EIDs] if self.edge_weights is not None else None
        )
        x = self.node_features[NIDs] if self.node_features is not None else None  # [32, 128]

        subg_aug = subg.add_self_loop()
        if edge_weights is not None:  # False
            edge_weights = torch.cat(
                [
                    edge_weights,
                    torch.ones(subg_aug.num_edges() - subg.num_edges()),
                ]
            )

        # compute structure from pyg data
        subg.x = x
        subg.z = z
        subg.node_id = NIDs
        subg.edge_index = torch.cat([subg.edges()[0].unsqueeze(0), subg.edges()[1].unsqueeze(0)], 0)
        if self.preprocess_fn is not None:
            self.preprocess_fn(subg, directed=self.directed, degree=None)

        return subg_aug, z, x, edge_weights, y, subg

    @property
    def cached_name(self):
        return f"SEAL_{self.split}_{self.percent}%.pt"

    def process(self):
        g_list, labels = [], []
        self.dynamic = True
        for i in tqdm(range(len(self))):
            g, z, x, weights, y = self[i]
            g.ndata["z"] = z
            if x is not None:
                g.ndata["x"] = x
            if weights is not None:
                g.edata["w"] = weights
            g_list.append(g)
            labels.append(y)
        self.dynamic = False
        return g_list, {"y": torch.tensor(labels)}

    def load_cached(self):
        path = os.path.join(self.root, self.cached_name)
        if os.path.exists(path):
            return load_graphs(path)

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        g_list, labels = self.process()
        save_graphs(path, g_list, labels)
        return g_list, labels


def train(num_datas):
    model.train()

    y_pred, y_true = torch.zeros([num_datas]), torch.zeros([num_datas])
    start = 0
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        g, z, x, edge_weights, y = [
            item.to(device) if item is not None else None for item in data
        ]
        g.attn_bias = g.attn_bias.to(device)
        g.edge_index = g.edge_index.to(device)
        g.x = g.x.to(device)
        g.z = g.z.to(device)
        if args.use_len_spd:
            g.len_shortest_path = g.len_shortest_path.to(device)
        if args.use_num_spd:
            g.num_shortest_path = g.num_shortest_path.to(device)
        if args.use_cnb_jac:
            g.undir_jac = g.undir_jac.to(device)
        if args.use_cnb_aa:
            g.undir_aa = g.undir_aa.to(device)
        if args.use_cnb_ra:
            g.undir_ra = g.undir_ra.to(device)
        if args.use_degree:
            g.undir_degree = g.undir_degree.to(device)
            if directed:
                g.in_degree = g.in_degree.to(device)
                g.out_degree = g.out_degree.to(device)

        num_datas_in_batch = y.numel()
        optimizer.zero_grad()
        logits = model(g, z, x, edge_weight=edge_weights)
        loss = BCEWithLogitsLoss()(logits.view(-1), y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * g.batch_size
        end = min(start+num_datas_in_batch, num_datas)
        y_pred[start:end] = logits.view(-1).cpu().detach()
        y_true[start:end] = y.view(-1).cpu().to(torch.float)
        start = end

    result = {}
    result['AUC'] = roc_auc_score(y_true, y_pred)
    return total_loss / len(train_dataset), result


def test_model(model, loader, num_datas):
    model.eval()

    y_pred, y_true = torch.zeros([num_datas]), torch.zeros([num_datas])
    start = 0
    for data in tqdm(loader, ncols=70):
        g, z, x, edge_weights, y = [
            item.to(device) if item is not None else None for item in data
        ]
        g.attn_bias = g.attn_bias.to(device)
        g.edge_index = g.edge_index.to(device)
        g.x = g.x.to(device)
        g.z = g.z.to(device)
        if args.use_len_spd:
            g.len_shortest_path = g.len_shortest_path.to(device)
        if args.use_num_spd:
            g.num_shortest_path = g.num_shortest_path.to(device)
        if args.use_cnb_jac:
            g.undir_jac = g.undir_jac.to(device)
        if args.use_cnb_aa:
            g.undir_aa = g.undir_aa.to(device)
        if args.use_cnb_ra:
            g.undir_ra = g.undir_ra.to(device)
        if args.use_degree:
            g.undir_degree = g.undir_degree.to(device)
            if directed:
                g.in_degree = g.in_degree.to(device)
                g.out_degree = g.out_degree.to(device)

        num_datas_in_batch = y.numel()
        logits = model(g, z, x, edge_weight=edge_weights)
        end = min(start+num_datas_in_batch, num_datas)
        y_pred[start:end] = logits.view(-1).cpu()
        y_true[start:end] = y.view(-1).cpu().to(torch.float)
        start = end

    pos_test_pred = y_pred[y_true==1]
    neg_test_pred = y_pred[y_true==0]
    return y_pred, y_true, pos_test_pred, neg_test_pred


def eval_model(**kwargs):
    eval_metric = kwargs["eval_metric"]
    if eval_metric == 'hits':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'mrr':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'auc':
        val_pred = kwargs["val_pred"]
        val_true = kwargs["val_true"]
        test_pred = kwargs["test_pred"]
        test_true = kwargs["test_true"]
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results

@torch.no_grad()
def test(eval_metric):
    model.eval()

    val_pred, val_true, pos_val_pred, neg_val_pred = test_model(model, val_loader, len(val_dataset))

    test_pred, test_true, pos_test_pred, neg_test_pred = test_model(model, test_loader, len(test_dataset))

    result = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                      val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric=eval_metric)
    if eval_metric != 'auc':
        result_auc = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                          val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric='auc')
        for key in result_auc.keys():
            result[key] = result_auc[key]
    return result

@torch.no_grad()
def final_test(eval_metric):
    model.eval()

    val_pred, val_true, pos_val_pred, neg_val_pred = test_model(model, final_val_loader, len(final_val_dataset))

    test_pred, test_true, pos_test_pred, neg_test_pred = test_model(model, final_test_loader, len(final_test_dataset))

    result = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                      val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric=eval_metric)
    if eval_metric != 'auc':
        result_auc = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                          val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric='auc')
        for key in result_auc.keys():
            result[key] = result_auc[key]
    return result


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)

    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results

# Data settings
parser = argparse.ArgumentParser(description='OGBL (SEAL)')
parser.add_argument('--cmd_time', type=str, default='ignore_time')
parser.add_argument('--root', type=str, default='dataset',
                    help="root of dataset")
parser.add_argument('--dataset', type=str, default='ogbl-collab')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")
# GNN settings
parser.add_argument('--model', type=str, default='SIEG')
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)

# Subgraph extraction settings
parser.add_argument('--sample_type', type=int, default=0)
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true',
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_feature_GT', action='store_true',
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")
parser.add_argument('--use_rpe', action='store_true', help="whether to use RPE as GNN input")
parser.add_argument('--replacement', action='store_true', help="whether to enable replacement sampleing in random walk")
parser.add_argument('--trackback', action='store_true', help="whether to enabale trackback path searching in random walk")
parser.add_argument('--num_walk', type=int, default=200, help='total number of random walks')
parser.add_argument('--num_step', type=int, default=4, help='total steps of random walk')
parser.add_argument('--rpe_hidden_dim', type=int, default=16, help='dimension of RPE embedding')
parser.add_argument('--gravity_type', type=int, default=0)
parser.add_argument('--readout_type', type=int, default=0)

# Training settings
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--train_percent', type=float, default=2)
parser.add_argument('--val_percent', type=float, default=1)
parser.add_argument('--test_percent', type=float, default=1)
parser.add_argument('--final_val_percent', type=float, default=1)
parser.add_argument('--final_test_percent', type=float, default=1)
parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_val', action='store_true')
parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument('--slice_type', type=int, default=0,
                    help="type of saving sampled subgraph in disk")
parser.add_argument('--num_workers', type=int, default=16, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true',
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--dont_z_emb_agg', action='store_true')
parser.add_argument('--pretrained_node_embedding', type=str, default=None, 
                    help="load pretrained node embeddings as additional node features")

# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix', type=str, default='', 
                    help="an appendix to the save directory")
parser.add_argument('--keep_old', action='store_true', 
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--only_test', action='store_true', 
                    help="only test without training")
parser.add_argument('--test_multiple_models', type=str, nargs='+', default=[], 
                    help="test multiple models together")
parser.add_argument('--use_heuristic', type=str, default=None,
                    help="test a link prediction heuristic (CN or AA)")
parser.add_argument('--num_heads', type=int, default=32)
parser.add_argument('--use_len_spd', action='store_true', default=False)
parser.add_argument('--use_num_spd', action='store_true', default=False)
parser.add_argument('--use_cnb_jac', action='store_true', default=False)
parser.add_argument('--use_cnb_aa', action='store_true', default=False)
parser.add_argument('--use_cnb_ra', action='store_true', default=False)
parser.add_argument('--use_degree', action='store_true', default=False)
parser.add_argument('--grpe_cross', action='store_true', default=False)
parser.add_argument('--use_ignn', action='store_true', default=False)
parser.add_argument('--mul_bias', action='store_true', default=False,
                    help="add bias to attention if true else multiple")
parser.add_argument('--max_z', type=int, default=1000)  # set a large max_z so that every z has embeddings to look up

parser.add_argument(
    "--ngnn_type",
    type=str,
    default="all",
    choices=["none", "input", "hidden", "output", "all"],
    help="You can set this value from 'none', 'input', 'hidden' or 'all' " \
            "to apply NGNN to different GNN layers.",
)
parser.add_argument(
    "--num_ngnn_layers", type=int, default=2, choices=[1, 2]
)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument(
    "--test_topk",
    type=int,
    default=1,
    help="select best k models for full validation/test each run.",
)
parser.add_argument(
    "--eval_hits_K",
    type=int,
    nargs="*",
    default=[10],
    help="hits@K for each eval step; " \
            "only available for datasets with hits@xx as the eval metric",
)
args = parser.parse_args()

if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = None if args.max_nodes_per_hop < 0 else args.max_nodes_per_hop
if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
    if args.use_valedges_as_input:
        args.data_appendix += '_uvai'
if args.use_heuristic is not None:
    args.runs = 1

args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
if not args.keep_old:
    # Backup files.
    backup_root_dir = os.path.join(args.res_dir, 'src')
    root_dir = os.path.dirname(sys.argv[0])
    root_dir = './' if root_dir == '' else root_dir
    for sub_dir in ['']:
        full_dir = os.path.join(root_dir, sub_dir)
        files = [f for f in os.listdir(full_dir) if os.path.isfile(os.path.join(full_dir, f)) and os.path.splitext(f)[1] in ['.py", ".c", ".cpp']]
        backup_dir = os.path.join(backup_root_dir, sub_dir)
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        for f in files:
            shutil.copy(os.path.join(full_dir, f), backup_dir)
log_file = os.path.join(args.res_dir, 'log.txt')
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
with open(log_file, 'a') as f:
    f.write('\n' + cmd_input)

if args.dataset.startswith('ogbl-citation'):
    args.eval_metric = 'mrr'
    directed = True
elif args.dataset.startswith('ogbl-vessel'):
    args.eval_metric = 'auc'
    directed = False
elif args.dataset.startswith('ogbl'):
    args.eval_metric = 'hits'
    directed = False
else:  # assume other datasets are undirected
    args.eval_metric = 'auc'
    directed = False

if args.dataset.startswith('ogbl'):
    evaluator = Evaluator(name=args.dataset)
if args.eval_metric == 'hits':
    loggers = {
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }
elif args.eval_metric == 'mrr':
    loggers = {
        'MRR': Logger(args.runs, args),
    }
elif args.eval_metric == 'auc':
    loggers = {
        'AUC': Logger(args.runs, args),
    }

device = 'cpu' if args.device == -1 or not torch.cuda.is_available() else f'cuda:{args.device}'
device = torch.device(device)

preprocess_fn = partial(preprocess,
                        grpe_cross=args.grpe_cross,
                        use_len_spd=args.use_len_spd,
                        use_num_spd=args.use_num_spd,
                        use_cnb_jac=args.use_cnb_jac,
                        use_cnb_aa=args.use_cnb_aa,
                        use_cnb_ra=args.use_cnb_ra,
                        use_degree=args.use_degree,
                        gravity_type=args.gravity_type,
                )

if args.dataset.startswith('ogbl'):
    dataset = DglLinkPropPredDataset(name=args.dataset)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    # Re-format the data of ogbl-citation2.
    if args.dataset == "ogbl-citation2":
        for k in ["train", "valid", "test"]:
            src = split_edge[k]["source_node"]
            tgt = split_edge[k]["target_node"]
            split_edge[k]["edge"] = torch.stack([src, tgt], dim=1)
            if k != "train":
                tgt_neg = split_edge[k]["target_node_neg"]
                split_edge[k]["edge_neg"] = torch.stack(
                    [src[:, None].repeat(1, tgt_neg.size(1)), tgt_neg], dim=-1
                )  # [Ns, Nt, 2]

    # Reconstruct the graph for ogbl-collab data 
    # for validation edge augmentation and coalesce.
    if args.dataset == "ogbl-collab":
        # Float edata for to_simple transformation.
        graph.edata.pop("year")
        graph.edata["weight"] = graph.edata["weight"].to(torch.float)
        if args.use_valedges_as_input:
            val_edges = split_edge["valid"]["edge"]
            row, col = val_edges.t()
            val_weights = torch.ones(size=(val_edges.size(0), 1))
            graph.add_edges(
                torch.cat([row, col]),
                torch.cat([col, row]),
                {"weight": val_weights},
            )
        graph = graph.to_simple(copy_edata=True, aggregator="sum")

    if args.dataset == "ogbl-vessel":
        graph.ndata["feat"][:, 0] = torch.nn.functional.normalize(
            graph.ndata["feat"][:, 0], dim=0
        )
        graph.ndata["feat"][:, 1] = torch.nn.functional.normalize(
            graph.ndata["feat"][:, 1], dim=0
        )
        graph.ndata["feat"][:, 2] = torch.nn.functional.normalize(
            graph.ndata["feat"][:, 2], dim=0
        )
        graph.ndata["feat"] = graph.ndata["feat"].to(torch.float)

    if not args.use_edge_weight and "weight" in graph.edata:
        del graph.edata["weight"]
    if not args.use_feature and "feat" in graph.ndata:
        del graph.ndata["feat"]

    data_appendix = "_rph{}".format("".join(str(args.ratio_per_hop).split(".")))
    path = f"{dataset.root}_seal{data_appendix}"
    if not (args.dynamic_train or args.dynamic_val or args.dynamic_test):
        args.num_workers = 0

    dataset_pyg = PygLinkPropPredDataset(name=args.dataset, root=args.root)
    data_pyg = dataset_pyg[0]
    train_dataset, val_dataset, test_dataset, final_val_dataset, final_test_dataset = [
        SEALOGBLDataset(
            data_pyg,
            preprocess_fn,
            path,
            graph,
            split_edge,
            percent=percent,
            split=split,
            ratio_per_hop=args.ratio_per_hop,
            directed=directed,
            dynamic=dynamic,
        )
        for percent, split, dynamic in zip(
            [
                args.train_percent,
                args.val_percent,
                args.test_percent,
                args.final_val_percent,
                args.final_test_percent,
            ],
            ["train", "valid", "test", "valid", "test"],
            [
                args.dynamic_train,
                args.dynamic_val,
                args.dynamic_test,
                args.dynamic_val,
                args.dynamic_test,
            ],
        )
    ]

    def ogbl_collate_fn(batch):
        gs, zs, xs, ws, ys, g_noaugs = zip(*batch)
        batched_g = dgl.batch(gs)
        z = torch.cat(zs, dim=0)
        if xs[0] is not None:
            x = torch.cat(xs, dim=0)
        else:
            x = None
        if ws[0] is not None:
            edge_weights = torch.cat(ws, dim=0)
        else:
            edge_weights = None
        y = torch.tensor(ys)

        # 把pairwise特征组装成batch
        batched_g.attn_bias = torch.cat([g_noaug.pair_attn_bias for g_noaug in g_noaugs], dim=0)
        batched_g.edge_index = torch.cat([g_noaug.pair_edge_idx for g_noaug in g_noaugs], dim=0)
        batched_g.x = torch.cat([g_noaug.pair_x for g_noaug in g_noaugs], dim=0)
        batched_g.z = torch.cat([g_noaug.pair_z for g_noaug in g_noaugs], dim=0)
        if args.use_len_spd:
            batched_g.len_shortest_path = torch.cat([g_noaug.pair_len_shortest_path for g_noaug in g_noaugs], dim=0)
        if args.use_num_spd:
            batched_g.num_shortest_path = torch.cat([g_noaug.pair_num_shortest_path for g_noaug in g_noaugs], dim=0)
        if args.use_cnb_jac:
            batched_g.undir_jac = torch.cat([g_noaug.pair_undir_jac for g_noaug in g_noaugs], dim=0)
        if args.use_cnb_aa:
            batched_g.undir_aa = torch.cat([g_noaug.pair_undir_aa for g_noaug in g_noaugs], dim=0)
        if args.use_cnb_ra:
            batched_g.undir_ra = torch.cat([g_noaug.pair_undir_ra for g_noaug in g_noaugs], dim=0)
        if args.use_degree:
            batched_g.undir_degree = torch.cat([g_noaug.pair_undir_degree for g_noaug in g_noaugs], dim=0)
            if directed:
                batched_g.in_degree = torch.cat([g_noaug.pair_in_degree for g_noaug in g_noaugs], dim=0)
                batched_g.out_degree = torch.cat([g_noaug.pair_out_degree for g_noaug in g_noaugs], dim=0)

        return batched_g, z, x, edge_weights, y

    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = GraphDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )
    final_val_loader = GraphDataLoader(
        final_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )
    final_test_loader = GraphDataLoader(
        final_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )

else:  # not ogbl-dataset:
    dataset = Planetoid(root='dataset', name=args.dataset)
    split_edge = do_edge_split(dataset, args.fast_split)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()


if args.train_node_embedding:
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
elif args.pretrained_node_embedding:
    weight = torch.load(args.pretrained_node_embedding)
    emb = torch.nn.Embedding.from_pretrained(weight)
    emb.weight.requires_grad=False
else:
    emb = None

if 0 < args.sortpool_k <= 1:  # Transform percentile to number.
    if args.dataset.startswith("ogbl-citation"):
        _sampled_indices = list(range(1000)) + list(
            range(len(train_dataset) - 1000, len(train_dataset))
        )
    else:
        _sampled_indices = list(range(1000))
    _num_nodes = sorted(
        [train_dataset[i][0].num_nodes() for i in _sampled_indices]
    )
    _k = _num_nodes[int(math.ceil(args.sortpool_k * len(_num_nodes))) - 1]
    model_k = max(10, _k)
else:
    model_k = int(args.sortpool_k)

print(f'args: {args}')
for run in range(args.runs):
    if args.model == 'SIEG':
        model = NGNNDGCNNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z,
                k=model_k, feature_dim=graph.ndata["feat"].size(1),
                use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                node_embedding=emb, readout_type=args.readout_type).to(device)

    print(model)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    lr = 2 * args.lr if args.scheduler else args.lr
    optimizer = torch.optim.Adam(params=parameters, lr=lr)  # , weight_decay=0.002
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)
    total_params = sum(p.numel() for param in parameters for p in param)
    localtime = time.asctime(time.localtime(time.time()))
    print(f'{localtime} Total number of parameters is {total_params}')
    print(f'SortPooling k is set to {model_k}')
    with open(log_file, 'a') as f:
        print(f'Total number of parameters is {total_params}', file=f)
        print(f'SortPooling k is set to {model_k}', file=f)

    start_epoch = 1
    # Training starts
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss, train_result = train(len(train_dataset))

        if epoch % args.eval_steps == 0:
            results = test(args.eval_metric)
            for key in loggers.keys():
                result = results[key]
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                model_name = os.path.join(
                    args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, epoch))
                optimizer_name = os.path.join(
                    args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run+1, epoch))
                torch.save(model.state_dict(), model_name)
                torch.save(optimizer.state_dict(), optimizer_name)

                train_str = []
                for key, result in train_result.items():
                    train_str.append(f'{key} {100 * result:.2f}%')
                valid_str = []
                test_str = []
                for key, result in results.items():
                    valid_res, test_res = result
                    valid_str.append(f'{key} {100 * valid_res:.2f}%')
                    test_str.append(f'{key} {100 * test_res:.2f}%')

                to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                            f'Loss: {loss:.4f}, {", ".join(train_str)}, ' +
                            f'Valid: {", ".join(valid_str)}, ' +
                            f'Test: {", ".join(test_str)}')
                localtime = time.asctime(time.localtime(time.time()))
                print(f'[{localtime}] {to_print}')
                with open(log_file, 'a') as f:
                    print(to_print, file=f)

    # choose the best model in valid for final_valid and final_test
    for key in loggers.keys():
        res = torch.tensor([val_test[0] for val_test in loggers[key].results[run]])
    idx_to_test = (
        torch.topk(res, 1, largest=True).indices + 1
    ).tolist()  # indices of top 1 valid results

    for _idx, epoch in enumerate(idx_to_test):
        model_name = os.path.join(
            args.res_dir, f"run{run+1}_model_checkpoint{epoch}.pth"
        )
        optimizer_name = os.path.join(
            args.res_dir,
            f"run{run+1}_optimizer_checkpoint{epoch}.pth",
        )
        model.load_state_dict(torch.load(model_name))
        optimizer.load_state_dict(torch.load(optimizer_name))

        results = final_test(args.eval_metric)
        for key in loggers.keys():
            result = results[key]
            loggers[key].add_result(run, result)

            final_valid_str = []
            final_test_str = []
            for key, result in results.items():
                final_valid_res, final_test_res = result
                final_valid_str.append(f'{key} {100 * final_valid_res:.2f}%')
                final_test_str.append(f'{key} {100 * final_test_res:.2f}%')

            to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                        f'Final Valid: {", ".join(final_valid_str)}, ' +
                        f'Final Test: {", ".join(final_test_str)}')
            localtime = time.asctime(time.localtime(time.time()))
            print(f'[{localtime}] {to_print}')
            with open(log_file, 'a') as f:
                print(to_print, file=f)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(run)
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(run, f=f)
        print(f'runs 0-{run}')
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(f'runs 0-{run}', file=f)
            loggers[key].print_statistics(f=f)

for key in loggers.keys():
    print(key)
    loggers[key].print_statistics()
    with open(log_file, 'a') as f:
        print(key, file=f)
        loggers[key].print_statistics(f=f)
print(f'Total number of parameters is {total_params}')
print(f'Results are saved in {args.res_dir}')
