import argparse
import time
import os, sys
import os.path as osp
from shutil import copy
import copy as cp
from tqdm import tqdm
from functools import partial
import psutil
import pdb

import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.sparse as ssp
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, IterableDataset

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from torch_geometric.datasets import Planetoid
from dataset import SEALDynamicDataset, SEALIterableDataset, SEALDynamicDataset
from preprocess import preprocess
from utils import *
from models import *
from timer_guard import TimerGuard

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', 
                    stream=sys.stdout,
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

def train():
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        new_data = data.clone()
        new_data.x = x
        new_data.edge_weight = edge_weight
        new_data.node_id = node_id
        logits = model(new_data)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / num_train_datas


def test_model(model, loader, num_datas):
    model.eval()

    y_pred, y_true = torch.zeros([num_datas]), torch.zeros([num_datas])
    start = 0
    for data in tqdm(loader, ncols=70):
        data = data.to(device)
        num_datas_in_batch = data.y.numel()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        new_data = data.clone()
        new_data.x = x
        new_data.edge_weight = edge_weight
        new_data.node_id = node_id
        logits = model(new_data)
        end = min(start+num_datas_in_batch, num_datas)
        y_pred[start:end] = logits.view(-1).cpu()
        y_true[start:end] = data.y.view(-1).cpu().to(torch.float)
        start = end
    pos_test_pred = y_pred[y_true==1]
    neg_test_pred = y_pred[y_true==0]

    return y_pred, y_true, pos_test_pred, neg_test_pred


def eval_model(**kwargs):
    if args.eval_metric == 'hits':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        val_pred = kwargs["val_pred"]
        val_true = kwargs["val_true"]
        test_pred = kwargs["test_pred"]
        test_true = kwargs["test_true"]
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results


@torch.no_grad()
def test():
    model.eval()

    val_pred, val_true, pos_val_pred, neg_val_pred = test_model(model, val_loader, num_val_datas)

    test_pred, test_true, pos_test_pred, neg_test_pred = test_model(model, test_loader, num_test_datas)

    return eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                      val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true)


@torch.no_grad()
def test_multiple_models_origin(models):
    num_models = len(models)
    for m in models:
        m.eval()

    y_preds, y_trues = [[] for _ in range(num_models)], [[] for _ in range(num_models)]
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, model in enumerate(models):
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_preds[i].append(logits.view(-1).cpu())
            y_trues[i].append(data.y.view(-1).cpu().to(torch.float))
    val_preds = [torch.cat(y_preds[i]) for i in range(num_models)]
    val_trues = [torch.cat(y_trues[i]) for i in range(num_models)]
    pos_val_preds = [val_preds[i][val_trues[i]==1] for i in range(num_models)]
    neg_val_preds = [val_preds[i][val_trues[i]==0] for i in range(num_models)]
    mem = psutil.virtual_memory()
    print(f' after val - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')

    y_preds, y_trues = [[] for _ in range(num_models)], [[] for _ in range(num_models)]
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, model in enumerate(models):
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_preds[i].append(logits.view(-1).cpu())
            y_trues[i].append(data.y.view(-1).cpu().to(torch.float))
    test_preds = [torch.cat(y_preds[i]) for i in range(num_models)]
    test_trues = [torch.cat(y_trues[i]) for i in range(num_models)]
    pos_test_preds = [test_preds[i][test_trues[i]==1] for i in range(num_models)]
    neg_test_preds = [test_preds[i][test_trues[i]==0] for i in range(num_models)]

    mem = psutil.virtual_memory()
    print(f' after test - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
    return eval_multiple_models(num_models,
                                pos_val_preds=pos_val_preds, neg_val_preds=neg_val_preds, pos_test_preds=pos_test_preds, neg_test_preds=neg_test_preds,
                                val_preds=val_preds, val_trues=val_trues, test_preds=test_preds, test_trues=test_trues)


@torch.no_grad()
def test_multiple_models(models, loader, num_datas):
    num_models = len(models)
    for m in models:
        m.eval()

    y_preds, y_trues = [torch.zeros([num_datas]) for _ in range(num_models)], [torch.zeros([num_datas]) for _ in range(num_models)]
    start = 0
    for data in tqdm(loader, ncols=70):
        data = data.to(device)
        num_datas_in_batch = data.y.numel()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        new_data = data.clone()
        new_data.x = x
        new_data.edge_weight = edge_weight
        new_data.node_id = node_id
        end = min(start+num_datas_in_batch, num_datas)
        for i, model in enumerate(models):
            logits = model(new_data)
            y_preds[i][start:end] = logits.view(-1).cpu()
            y_trues[i][start:end] = data.y.view(-1).cpu().to(torch.float)
        start = end
    pos_test_preds = [y_preds[i][y_trues[i]==1] for i in range(num_models)]
    neg_test_preds = [y_preds[i][y_trues[i]==0] for i in range(num_models)]

    mem = psutil.virtual_memory()
    print(f'       max - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
    return y_preds, y_trues, pos_test_preds, neg_test_preds


def eval_multiple_models(num_models, **kwargs):
    Results = []
    for i in range(num_models):
        if args.eval_metric == 'hits':
            pos_val_preds = kwargs["pos_val_preds"]
            neg_val_preds = kwargs["neg_val_preds"]
            pos_test_preds = kwargs["pos_test_preds"]
            neg_test_preds = kwargs["neg_test_preds"]
            Results.append(evaluate_hits(pos_val_preds[i], neg_val_preds[i], pos_test_preds[i], neg_test_preds[i]))
        elif args.eval_metric == 'mrr':
            pos_val_preds = kwargs["pos_val_preds"]
            neg_val_preds = kwargs["neg_val_preds"]
            pos_test_preds = kwargs["pos_test_preds"]
            neg_test_preds = kwargs["neg_test_preds"]
            Results.append(evaluate_mrr(pos_val_preds[i], neg_val_preds[i], pos_test_preds[i], neg_test_preds[i]))
        elif args.eval_metric == 'auc':
            val_preds = kwargs["val_preds"]
            val_trues = kwargs["val_trues"]
            test_preds = kwargs["test_preds"]
            test_trues = kwargs["test_trues"]
            Results.append(evaluate_auc(val_preds[i], val_trues[i], test_preds[i], test_true[i]))
    return Results


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
parser = argparse.ArgumentParser(description='OGBL')
parser.add_argument('--root', type=str, default='dataset',
                    help="root of dataset")
parser.add_argument('--dataset', type=str, default='ogbl-collab')
parser.add_argument('--fast_split', action='store_true', 
                    help="for large custom datasets (not OGB), do a fast data split")
# GNN settings
parser.add_argument('--model', type=str, default='DGCNN')
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
parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")
parser.add_argument('--readout_type', type=int, default=0)
# Training settings
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--val_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)
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
parser.add_argument('--use_num_spd', action='store_true', default=False)
parser.add_argument('--use_cnb_jac', action='store_true', default=False)
parser.add_argument('--use_cnb_aa', action='store_true', default=False)
parser.add_argument('--use_degree', action='store_true', default=False)
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
    # Backup python files.
    copy('train.py', args.res_dir)
    copy('dataset.py', args.res_dir)
    copy('models.py', args.res_dir)
    copy('utils.py', args.res_dir)
log_file = os.path.join(args.res_dir, 'log.txt')
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
with open(log_file, 'a') as f:
    f.write('\n' + cmd_input)

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

if args.dataset.startswith('ogbl'):
    dataset = PygLinkPropPredDataset(name=args.dataset, root=args.root)
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    print(get_dict_info(split_edge))
    print(f'data {data}')
else:
    dataset = Planetoid(root='dataset', name=args.dataset)
    split_edge = do_edge_split(dataset, args.fast_split)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()
print(data)
for k, v in split_edge.items():
    print(f'split_edge[\'{k}\']: {list(v.keys())}')

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

if args.use_valedges_as_input:
    val_edge_index = split_edge['valid']['edge'].t()
    if not directed:
        val_edge_index = to_undirected(val_edge_index)
    data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
    data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

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
    
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:0')
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

if args.use_heuristic:
    # Test link prediction heuristics.
    num_nodes = data.num_nodes
    if 'edge_weight' in data and args.use_edge_weight:
        edge_weight = data.edge_weight.view(-1)
    else:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), 
                       shape=(num_nodes, num_nodes))

    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge, 
                                                   data.edge_index, 
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge, 
                                                     data.edge_index, 
                                                     data.num_nodes)
    if directed:
        cn_types = ['undirected', 'in', 'out', 's2o', 'o2s']
    else:
        cn_types = ['in']
    for cn_type in cn_types:
        pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge, cn_type=cn_type)
        neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge, cn_type=cn_type)
        pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge, cn_type=cn_type)
        neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge, cn_type=cn_type)

        if args.eval_metric == 'hits':
            results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
        elif args.eval_metric == 'mrr':
            results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
        elif args.eval_metric == 'auc':
            val_pred = torch.cat([pos_val_pred, neg_val_pred])
            val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                                  torch.zeros(neg_val_pred.size(0), dtype=int)])
            test_pred = torch.cat([pos_test_pred, neg_test_pred])
            test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                                  torch.zeros(neg_test_pred.size(0), dtype=int)])
            results = evaluate_auc(val_pred, val_true, test_pred, test_true)

        for key, result in results.items():
            loggers[key].reset()
            loggers[key].add_result(0, result)
        for key in loggers.keys():
            print(cn_type)
            print(key)
            loggers[key].print_statistics()
            with open(log_file, 'a') as f:
                print(cn_type, file=f)
                print(key, file=f)
                loggers[key].print_statistics(f=f)
    pdb.set_trace()
    exit()


path = dataset.root + '_{}'.format(args.data_appendix)
print(f'path {path}')
use_coalesce = True if args.dataset == 'ogbl-collab' else False
#if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
#    args.num_workers = 0

preprocess_fn = partial(preprocess,
                        use_num_spd=args.use_num_spd,
                        use_cnb_jac=args.use_cnb_jac,
                        use_cnb_aa=args.use_cnb_aa,
                        use_degree=args.use_degree,
                ) if args.model.find('Graphormer') != -1 else None

dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALIterableDataset'
train_dataset = eval(dataset_class)(
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.train_percent, 
    split='train', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    directed=directed, 
    sample_type=args.sample_type,
    shuffle=True,
    slice_type=args.slice_type,
    preprocess_fn=preprocess_fn,
) 
num_train_datas = len(train_dataset)
if False:  # visualize some graphs
    import networkx as nx
    from torch_geometric.utils import to_networkx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for g in loader:
        f = plt.figure(figsize=(20, 20))
        limits = plt.axis('off')
        g = g.to(device)
        node_size = 100
        with_labels = True
        G = to_networkx(g, node_attrs=['z'])
        labels = {i: G.nodes[i]['z'] for i in range(len(G))}
        nx.draw(G, node_size=node_size, arrows=True, with_labels=with_labels,
                labels=labels)
        f.savefig('tmp_vis.png')
        pdb.set_trace()

dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALIterableDataset'
val_dataset = eval(dataset_class)(
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.val_percent, 
    split='valid', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    directed=directed, 
    sample_type=args.sample_type,
    slice_type=args.slice_type,
    preprocess_fn=preprocess_fn,
)
num_val_datas = len(val_dataset)

dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALIterableDataset'
test_dataset = eval(dataset_class)(
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.test_percent, 
    split='test', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    directed=directed, 
    sample_type=args.sample_type,
    slice_type=args.slice_type,
    preprocess_fn=preprocess_fn,
)
num_test_datas = len(test_dataset)

max_z = 1000  # set a large max_z so that every z has embeddings to look up

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                          shuffle=True if args.dynamic_train else False,
                          num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                        num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                         num_workers=args.num_workers)

if args.train_node_embedding:
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
elif args.pretrained_node_embedding:
    weight = torch.load(args.pretrained_node_embedding)
    emb = torch.nn.Embedding.from_pretrained(weight)
    emb.weight.requires_grad=False
else:
    emb = None

print(f'args: {args}')
for run in range(args.runs):
    if args.model == 'DGCNN':
        model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k, 
                      train_dataset, use_feature=args.use_feature, 
                      node_embedding=emb).to(device)
    elif args.model == 'SAGE':
        model = SAGE(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                     args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GCN':
        model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GIN':
        model = GIN(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GCNGraphormer':
        model = GCNGraphormer(args, args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature, node_embedding=emb, args=args).to(device)
    elif args.model == 'DGCNNGraphormer':
        model = DGCNNGraphormer(args, args.hidden_channels, args.num_layers, max_z, args.sortpool_k, 
                      train_dataset, use_feature=args.use_feature, 
                      node_embedding=emb, readout_type=args.readout_type).to(device)

    print(model)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    total_params = sum(p.numel() for param in parameters for p in param)
    localtime = time.asctime(time.localtime(time.time()))
    print(f'{localtime} Total number of parameters is {total_params}')
    if args.model.find('DGCNN') != -1:
        print(f'SortPooling k is set to {model.k}')
    with open(log_file, 'a') as f:
        print(f'Total number of parameters is {total_params}', file=f)
        if args.model.find('DGCNN') != -1:
            print(f'SortPooling k is set to {model.k}', file=f)

    start_epoch = 1
    if args.continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_model_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_optimizer_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        start_epoch = args.continue_from + 1
        args.epochs -= args.continue_from
    
    if args.only_test:
        results = test()
        for key, result in results.items():
            loggers[key].add_result(run, result)
        for key, result in results.items():
            valid_res, test_res = result
            localtime = time.asctime(time.localtime(time.time()))
            print(f'[{localtime}] {key}')
            print(f'[{localtime}] Run: {run + 1:02d}, '
                  f'[{localtime}] Valid: {100 * valid_res:.2f}%, '
                  f'[{localtime}] Test: {100 * test_res:.2f}%')
        pdb.set_trace()
        exit()

    if len(args.test_multiple_models) > 0 :
        model_paths = args.test_multiple_models
        num_models = len(model_paths)
        models = []
        for path in model_paths:
            m = cp.deepcopy(model)
            m.load_state_dict(torch.load(path))
            models.append(m)
        print(f'       tag - percent - mem.free(Gb) - mem.avail(Gb) - mem.used(Gb)')
        mem = psutil.virtual_memory()
        print(f'     begin - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
        if args.eval_metric != 'auc':
            _, _, pos_val_preds, neg_val_preds = test_multiple_models(models, val_loader, num_val_datas)
        else:
            val_preds, val_trues, _, _ = test_multiple_models(models, val_loader, num_val_datas)
        mem = psutil.virtual_memory()
        print(f' after val - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
        if args.eval_metric != 'auc':
            _, _, pos_test_preds, neg_test_preds = test_multiple_models(models, test_loader, num_test_datas)
        else:
            test_preds, test_trues, _, _ = test_multiple_models(models, test_loader, num_test_datas)
        mem = psutil.virtual_memory()
        print(f'after test - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
        if args.eval_metric != 'auc':
            Results = eval_multiple_models(num_models,
                                 pos_val_preds=pos_val_preds, neg_val_preds=neg_val_preds,
                                 pos_test_preds=pos_test_preds, neg_test_preds=neg_test_preds)
        else:
            Results = eval_multiple_models(num_models,
                                 val_preds=val_preds, val_trues=val_trues, test_preds=test_preds, test_trues=test_trues)
        mem = psutil.virtual_memory()
        print(f'     final - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
        for i, path in enumerate(model_paths):
            print(path)
            with open(log_file, 'a') as f:
                print(path, file=f)
            results = Results[i]
            for key, result in results.items():
                loggers[key].add_result(run, result)
            for key, result in results.items():
                valid_res, test_res = result
                to_print = (f'Run: {run + 1:02d}, ' +
                            f'Valid: {100 * valid_res:.2f}%, ' +
                            f'Test: {100 * test_res:.2f}%')
                localtime = time.asctime(time.localtime(time.time()))
                print(f'[{localtime}] {key}')
                print(f'[{localtime}] {to_print}')
                with open(log_file, 'a') as f:
                    print(key, file=f)
                    print(to_print, file=f)
        for key, result in results.items():
            print(key)
            loggers[key].print_statistics(run, std=True)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f, std=True)
        pdb.set_trace()
        exit()

    # Training starts
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss = train()

        if epoch % args.eval_steps == 0:
            results = test()
            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                model_name = os.path.join(
                    args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, epoch))
                optimizer_name = os.path.join(
                    args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run+1, epoch))
                torch.save(model.state_dict(), model_name)
                torch.save(optimizer.state_dict(), optimizer_name)

                for key, result in results.items():
                    valid_res, test_res = result
                    to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                                f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                                f'Test: {100 * test_res:.2f}%')
                    localtime = time.asctime(time.localtime(time.time()))
                    print(f'[{localtime}] {key}')
                    print(f'[{localtime}] {to_print}')
                    with open(log_file, 'a') as f:
                        print(key, file=f)
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


