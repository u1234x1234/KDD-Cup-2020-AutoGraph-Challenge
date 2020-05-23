# %%
import sys
sys.path.append('/home/u1234x1234/autograph2020/src')
from data_utils import read_dataset
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


from torch_geometric.nn import GatedGraphConv, GATConv, TAGConv, ChebConv, GCNConv, SAGEConv
from uxils.automl.parametric_family import ParametricFamilyModel
from uxils.ray_ext import optimize_function, show_results
import numpy as np
from uxils.functools_ext import partialclass
from ag.pyg_model import PYGModel
from model import Model
from sklearn.metrics import accuracy_score
from uxils.timer import Timer
from ag.pyg_utils import generate_pyg_data
from sklearn.decomposition import TruncatedSVD, PCA
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


dataset, y_test = read_dataset('b')
n_class = dataset.get_metadata()['n_class']
schema = dataset.get_metadata()['schema']
time_budget = dataset.get_metadata()['time_budget']

# %%

def compute_nn_features(X, edges_list):
    edges_to = defaultdict(list)
    edges_from = defaultdict(list)

    for u, v in edges_list:
        edges_to[u].append(v)
        edges_from[v].append(u)

    X_n1_to_mean = []
    X_n1_to_sum = []
    X_n1_from_mean = []
    X_n2_to_mean = []
    dim = X.shape[1]
    X_hc = []
    X_n1_to_from_mean = []
    from scipy.stats import gmean

    for node in range(X.shape[0]):
        n1_to = [v for v in edges_to[node]] + [node]
        X_n1_to = np.array([X[v] for v in n1_to])
        if len(n1_to) == 0:
            n1_to = np.zeros((1, dim))
        X_n1_to_mean.append(np.median(X_n1_to, axis=0))
        X_n1_to_sum.append(X_n1_to.sum(axis=0))

        n1_from = np.array([X[u] for u in edges_from[node]])
        if len(n1_from) == 0:
            n1_from = np.zeros((1, dim))
        X_n1_from_mean.append(np.median(n1_from, axis=0))

        n1_to_from = [v for v in edges_to[node]] + [v for v in edges_from[node]] + [node]
        X_n1_to_from = np.array([X[v] for v in n1_to_from])
        X_n1_to_from_mean.append(np.mean(X_n1_to_from, axis=0))

        n2_to = set(v2 for v in edges_to[node] for v2 in edges_to[v])
        for v in edges_to[node]:
            n2_to.add(v)
        n2_to.add(node)
        n2_to = np.array([X[v] for v in n2_to])
        X_n2_to_mean.append(np.mean(n2_to, axis=0))
        X_hc.append([len(edges_from[node]), len(edges_to[node])])

    X_n1_to_mean = np.array(X_n1_to_mean)
    X_n1_to_sum = np.array(X_n1_to_sum)
    X_n1_from_mean = np.array(X_n1_from_mean)
    X_hc = np.array(X_hc)

    X = np.hstack([X, X_n1_to_mean, X_n1_to_sum, X_n1_to_from_mean, X_n2_to_mean])
    # X = StandardScaler().fit_transform(X)

    return X


# %%

pyg_data = generate_pyg_data(dataset.get_data())
print(pyg_data)
X = pyg_data.x.numpy()
X = compute_nn_features(X, pyg_data.edge_index.t().tolist())

X = StandardScaler().fit_transform(X)

X_train = X[dataset.get_data()['train_indices']]
X_test = X[dataset.get_data()['test_indices']]

y_train = dataset.get_data()['train_label'].label.to_numpy()
assert len(y_train) == len(X_train)
assert len(X_test) == len(y_test)
print(X_train.shape, X_test.shape)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from cleanlab.pruning import get_noise_indices
from cleanlab.classification import LearningWithNoisyLabels

model = LogisticRegression()
model = LearningWithNoisyLabels(model)

model.fit(X_train, y_train)
y_pred = np.array(model.predict(X_test))

score = accuracy_score(y_test, y_pred)
print(score)


# %%


# m = Model()
# y_pred = m.train_predict(dataset.get_data(), time_budget=100, n_class=n_class, schema=schema)
# score = accuracy_score(y_test, y_pred)
# print(score)
# qwe
from torch_geometric.nn import SAGEConv, SplineConv, GraphConv, GravNetConv, GINConv, ARMAConv, SGConv, RGCNConv, FeaStConv


search_space = {
    'conv_class': [
        GCNConv,
        partialclass(GCNConv, improved=True),
        partialclass(GCNConv, normalize=True),

        partialclass(ChebConv, K=3),
        partialclass(ChebConv, K=7),

        SAGEConv,
        partialclass(SAGEConv, concat=True),
        partialclass(SAGEConv, normalize=True),

        partialclass(GraphConv, aggr='add'),
        partialclass(GraphConv, aggr='mean'),

        partialclass(GatedGraphConv, aggr='mean'),  # SWAP params
        partialclass(GatedGraphConv, aggr='add'),

        partialclass(GATConv, concat=True),
        partialclass(GATConv, dropout=0.5),

        partialclass(TAGConv, K=3),
        partialclass(TAGConv, K=5),
        partialclass(ARMAConv, dropout=0, num_layers=2, num_stacks=3),
        partialclass(ARMAConv, dropout=0, num_layers=3, num_stacks=2),
        partialclass(ARMAConv, dropout=0.5, num_layers=3, num_stacks=3),
        partialclass(ARMAConv, dropout=0.5),

        partialclass(SGConv, K=1),
        partialclass(SGConv, K=2),

        partialclass(FeaStConv, heads=1)
    ],
    'num_layers': [1, 2],
    'hidden_size': [32, 64],
    'in_dropout': [0.5, 0.8],
    'out_dropout': [0.5, 0.8],
    'n_iter': [300, 500, 700, 1000],
    'weight_decay': [0, 1e-4, 1e-3],
    'learning_rate': [0.01, 0.001]
}


def func(config):
    dataset, y_test = read_dataset('a')
    n_class = dataset.get_metadata()['n_class']
    pyg_data = generate_pyg_data(dataset.get_data())

    input_size = pyg_data.x.size(1)

    model = PYGModel(
        n_class, input_size, config['conv_class'], hidden_size=config['hidden_size'], num_layers=config['num_layers'],
        in_dropout=config['in_dropout'], out_dropout=config['out_dropout'], n_iter=config['n_iter'],
        weight_decay=config['weight_decay'], learning_rate=config['learning_rate'])
    y_pred, score = model.fit_predict(pyg_data, full=True)
    y_pred = y_pred.argmax(axis=1)
    score = accuracy_score(y_test, y_pred)
    return score


r = optimize_function(func, search_space, time_budget=600, num_gpus=2, num_cpus=20, cpu_per_trial=2, gpu_per_trial=0.5)
import joblib
joblib.dump(r, 'task_a.pkl')

show_results(r)


# %%

dataset, y_test = read_dataset('a')

n_class = dataset.get_metadata()['n_class']
schema = dataset.get_metadata()['schema']
time_budget = dataset.get_metadata()['time_budget']
pyg_data = generate_pyg_data(dataset.get_data())

from functools import partial
input_size = pyg_data.x.size(1)
print(input_size)
config = {
    'conv_class': partial(ChebConv, K=7),
    'hidden_size': 32,
    'num_layers': 2,
    'in_dropout': 0.5,
    'out_dropout': 0.5,
    'n_iter': 200,
}

model = PYGModel(
    n_class, input_size, config['conv_class'], hidden_size=config['hidden_size'], num_layers=config['num_layers'],
    in_dropout=config['in_dropout'], out_dropout=config['out_dropout'], n_iter=config['n_iter'])
y_pred, t_pred = model.fit_predict(pyg_data, full=True)
y_pred = y_pred.argmax(axis=1)
score = accuracy_score(y_test, y_pred)
print(score)

from scipy.special import softmax
t_pred = softmax(t_pred, axis=1)
test_mask = ~(pyg_data.train_mask + pyg_data.val_mask).cpu().numpy()
perc = np.percentile(t_pred[test_mask].max(axis=1), 15)
nmask = ((t_pred * test_mask[:, np.newaxis]).max(axis=1) > perc)

import torch
nmask = torch.tensor(nmask, dtype=torch.bool).cuda()
pyg_data.train_mask += nmask
pyg_data.y[test_mask] = torch.tensor(t_pred[test_mask].argmax(axis=1), dtype=torch.long).cuda()


model = PYGModel(
    n_class, input_size, config['conv_class'], hidden_size=config['hidden_size'], num_layers=config['num_layers'],
    in_dropout=config['in_dropout'], out_dropout=config['out_dropout'], n_iter=config['n_iter'])
y_pred, t_pred = model.fit_predict(pyg_data, full=True, m=nmask)

y_pred = y_pred.argmax(axis=1)
score = accuracy_score(y_test, y_pred)
print(score)
