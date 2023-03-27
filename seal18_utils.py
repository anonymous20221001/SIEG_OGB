from __future__ import print_function
import numpy as np
import random
import os, sys, pdb, math, time
import networkx as nx
import scipy.io as sio
import scipy.sparse as ssp
import sys, copy, math, time, pdb
import pickle
import os.path


def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None,
               all_unknown_as_negative=False):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    if not all_unknown_as_negative:
        # sample a portion unknown links as train_negs and test_negs (no overlap)
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        # regard all unknown links as test_negs, sample a portion from them as train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net==0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg


'''Prepare data'''
def seal18_prepare_data(args):
    file_dir = os.path.dirname(os.path.realpath('__file__'))

    # check whether train and test links are provided
    train_pos, test_pos = None, None
    if args.train_name is not None:
        train_dir = os.path.join(file_dir, 'seal18_data/{}'.format(args.train_name))
        train_idx = np.loadtxt(train_dir, dtype=int)
        train_pos = (train_idx[:, 0], train_idx[:, 1])
    if args.test_name is not None:
        test_dir = os.path.join(file_dir, 'seal18_data/{}'.format(args.test_name))
        test_idx = np.loadtxt(test_dir, dtype=int)
        test_pos = (test_idx[:, 0], test_idx[:, 1])

    # build observed network
    if args.dataset is not None:  # use .mat network
        data_dir = os.path.join(file_dir, 'seal18_data/{}.mat'.format(args.dataset))
        data = sio.loadmat(data_dir)
        net = data['net']
        if 'group' in data:
            # load node attributes (here a.k.a. node classes)
            attributes = data['group'].toarray().astype('float32')
        else:
            attributes = None
        # check whether net is symmetric (for small nets only)
        if False:
            net_ = net.toarray()
            assert(np.allclose(net_, net_.T, atol=1e-8))
    else:  # build network from train links
        assert (args.train_name is not None), "must provide train links if not using .mat"
        if args.train_name.endswith('_train.txt'):
            args.dataset = args.train_name[:-10] 
        else:
            args.dataset = args.train_name.split('.')[0]
        max_idx = np.max(train_idx)
        if args.test_name is not None:
            max_idx = max(max_idx, np.max(test_idx))
        net = ssp.csc_matrix(
            (np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), 
            shape=(max_idx+1, max_idx+1)
        )
        net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
        net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops

    # sample train and test links
    if args.train_name is None and args.test_name is None:
        # sample both positive and negative train/test links from net
        train_pos, train_neg, test_pos, test_neg = sample_neg(
            net, args.test_ratio, max_train_num=args.max_train_num
        )
    else:
        # use provided train/test positive links, sample negative from net
        train_pos, train_neg, test_pos, test_neg = sample_neg(
            net, 
            train_pos=train_pos, 
            test_pos=test_pos, 
            max_train_num=args.max_train_num,
            all_unknown_as_negative=args.all_unknown_as_negative
        )
    return train_pos, train_neg, test_pos, test_neg


# '''Train and apply classifier'''
# A = net.copy()  # the observed network
# A[test_pos[0], test_pos[1]] = 0  # mask test links
# A[test_pos[1], test_pos[0]] = 0  # mask test links
# A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

# node_information = None
# if args.use_embedding:
#     embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
#     node_information = embeddings
# if args.use_attribute and attributes is not None:
#     if node_information is not None:
#         node_information = np.concatenate([node_information, attributes], axis=1)
#     else:
#         node_information = attributes

# if args.only_predict:  # no need to use negatives
#     _, test_graphs, max_n_label = links2subgraphs(
#         A, 
#         None, 
#         None, 
#         test_pos, # test_pos is a name only, we don't actually know their labels
#         None, 
#         args.hop, 
#         args.max_nodes_per_hop, 
#         node_information, 
#         args.no_parallel
#     )
#     print('# test: %d' % (len(test_graphs)))
# else:
#     train_graphs, test_graphs, max_n_label = links2subgraphs(
#         A, 
#         train_pos, 
#         train_neg, 
#         test_pos, 
#         test_neg, 
#         args.hop, 
#         args.max_nodes_per_hop, 
#         node_information, 
#         args.no_parallel
#     )
#     print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))
# import pdb; pdb.set_trace()
