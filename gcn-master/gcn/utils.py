import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

'''
def isolate_nodes(adj, features, n, seed=42):
    """
    随机孤立n个节点：特征全为0，邻接矩阵对应行列全为0
    返回修改后的adj和features，以及被孤立的节点索引列表
    """
    np.random.seed(seed)
    num_nodes = features.shape[0]
    idx = np.random.choice(num_nodes, n, replace=False)
    # 特征置0
    features[idx, :] = 0
    # 邻接矩阵行列置0
    adj = adj.tolil()
    for i in idx:
        adj[i, :] = 0
        adj[:, i] = 0
    return adj.tocsr(), features, idx
'''

'''
def add_nodes(adj, features, n, m, feature_mode='random', seed=42):
    """
    随机增加n个节点到图中
    adj: scipy.sparse 矩阵
    features: scipy.sparse 矩阵
    n: 新增节点数
    m: 每个新节点的邻居数
    feature_mode: 'random' 或 'mean'
    返回新的adj和features，以及新节点的索引列表
    """
    np.random.seed(seed)
    num_nodes = features.shape[0]
    num_features = features.shape[1]

    # 生成新节点特征
    if feature_mode == 'random':
        new_features = np.random.rand(n, num_features)
    elif feature_mode == 'mean':
        new_features = features.mean(axis=0)
        if hasattr(new_features, 'A1'):  # 稀疏矩阵
            new_features = new_features.A1
        new_features = np.tile(new_features, (n, 1))
    else:
        raise ValueError('Unknown feature_mode')

    # 拼接特征
    if sp.issparse(features):
        features = features.tolil()
        features = sp.vstack([features, sp.csr_matrix(new_features)])
    else:
        features = np.vstack([features, new_features])

    # 扩展邻接矩阵为方阵
    adj = adj.tolil()
    adj_rows, adj_cols = adj.shape
    new_size = adj_rows + n
    adj.resize((new_size, new_size))

    # 新节点连边
    for i in range(n):
        new_idx = num_nodes + i
        neighbors = np.random.choice(num_nodes, m, replace=False)
        for nb in neighbors:
            adj[new_idx, nb] = 1
            adj[nb, new_idx] = 1

    return adj.tocsr(), features, list(range(num_nodes, num_nodes + n))
'''


def add_nodes(adj, features, n, m, feature_mode, neighbor_mode, seed=42):
    """
    随机或按度值增加n个节点到图中
    neighbor_mode: 'random' 或 'max_degree'
    """
    np.random.seed(seed)
    num_nodes = features.shape[0]
    num_features = features.shape[1]

    # 生成新节点特征
    if feature_mode == 'random':
        new_features = np.random.rand(n, num_features)
    elif feature_mode == 'mean':
        new_features = features.mean(axis=0)
        if hasattr(new_features, 'A1'):  # 稀疏矩阵
            new_features = new_features.A1
        new_features = np.tile(new_features, (n, 1))
    else:
        raise ValueError('Unknown feature_mode')

    # 拼接特征
    if sp.issparse(features):
        features = features.tolil()
        features = sp.vstack([features, sp.csr_matrix(new_features)])
    else:
        features = np.vstack([features, new_features])

    # 扩展邻接矩阵为方阵
    adj = adj.tolil()
    adj_rows, adj_cols = adj.shape
    new_size = adj_rows + n
    adj.resize((new_size, new_size))

    # 新节点连边
    for i in range(n):
        new_idx = num_nodes + i
        if neighbor_mode == 'random':
            neighbors = np.random.choice(num_nodes, m, replace=False)
        elif neighbor_mode == 'max_degree':
            degrees = np.array(adj[:num_nodes, :num_nodes].sum(axis=1)).flatten()
            neighbors = np.argsort(-degrees)[:m]  # 取度最大的m个节点
        else:
            raise ValueError('Unknown neighbor_mode')
        for nb in neighbors:
            adj[new_idx, nb] = 1
            adj[nb, new_idx] = 1

    return adj.tocsr(), features, list(range(num_nodes, num_nodes + n))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
