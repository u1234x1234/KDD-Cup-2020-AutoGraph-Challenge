# %%
import sys
sys.path.append('/home/u1234x1234/autograph2020/src')
from data_utils import read_dataset


from torch_geometric.nn import GatedGraphConv, GATConv, TAGConv, ChebConv, GCNConv
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

    for node in range(X.shape[0]):
        n1_to = [v for v in edges_to[node]] + [node]
        n1_to = np.array([X[v] for v in n1_to])
        if len(n1_to) == 0:
            n1_to = np.zeros((1, dim))
        X_n1_to_mean.append(np.median(n1_to, axis=0))
        # X_n1_to_mean.append(n1_to.mean(axis=0))
        X_n1_to_sum.append(n1_to.sum(axis=0))

        n1_from = np.array([X[u] for u in edges_from[node]])
        if len(n1_from) == 0:
            n1_from = np.zeros((1, dim))
        X_n1_from_mean.append(np.median(n1_from, axis=0))

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

    # X_n1_to_mean = StandardScaler().fit_transform(X_n1_to_mean)

    print(X.shape, X_n1_to_mean.shape, X_n1_to_sum.shape, X_n1_from_mean.shape)
    X = np.hstack([X, X_n1_to_mean, X_n1_to_sum, X_n2_to_mean, np.abs(X_n1_to_mean - X_n1_from_mean)])
    return X


# %%

dataset, y_test = read_dataset('a')
n_class = dataset.get_metadata()['n_class']
schema = dataset.get_metadata()['schema']
time_budget = dataset.get_metadata()['time_budget']

# %%
m = Model()
y_pred = m.train_predict(dataset.get_data(), time_budget=100, n_class=n_class, schema=schema)
score = accuracy_score(y_test, y_pred)
print(score)

qwe

with Timer('pyg'):
    pyg_data = generate_pyg_data(dataset.get_data())

X = dataset.get_data()['fea_table'].drop('node_index', axis=1).to_numpy()
print(X.shape)

X = compute_nn_features(X, pyg_data.edge_index.t().tolist())

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD
X = StandardScaler().fit_transform(X)

X_train = X[dataset.get_data()['train_indices']]
X_test = X[dataset.get_data()['test_indices']]

y_train = dataset.get_data()['train_label'].label.to_numpy()
assert len(y_train) == len(X_train)
assert len(X_test) == len(y_test)

print(X_train.shape, X_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# model = RandomForestClassifier(n_estimators=600, n_jobs=10)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)


qwe

# %%


search_space = {
    'conv_class': [
        GCNConv,
        partialclass(GCNConv, improved=True),
        partialclass(GCNConv, normalize=True),

        partialclass(ChebConv, K=3),
        # partialclass(ChebConv, K=7),

        # SAGEConv,
        # partialclass(SAGEConv, concat=True),
        # partialclass(SAGEConv, normalize=True),

        # partialclass(SplineConv, dim=3, kernel_size=5),
        # partialclass(GraphConv, aggr='mean'),
        # PCLS(GravNetConv, space_dimensions=16, propagate_dimensions=16, k=5),
        # partialclass(GatedGraphConv, aggr='mean'), partialclass(GatedGraphConv, aggr='add'),
        # TAGConv
    ],
    'num_layers': [2],
    'hidden_size': [128],
}


def func(config):
    dataset, y_test = read_dataset('b')

    # print(dataset.fea_table.values.shape)
    # pca = TruncatedSVD(n_components=550, algorithm='arpack')
    # X = pca.fit_transform(dataset.fea_table.values[:, 1:])
    # dataset.fea_table = pd.DataFrame(X)
    # dataset.fea_table['node_index'] = pd.Series(np.random.randint(10, size=len(dataset.fea_table)))

    n_class = dataset.get_metadata()['n_class']
    schema = dataset.get_metadata()['schema']
    time_budget = dataset.get_metadata()['time_budget']
    pyg_data = generate_pyg_data(dataset.get_data())

    import torch
    x = compute_nn_features(pyg_data.x.numpy(), pyg_data.edge_index.t().tolist())
    # x = StandardScaler().fit_transform(x)
    # from sklearn.decomposition import PCA
    x = PCA(n_components=512, svd_solver='arpack').fit_transform(x).copy()
    pyg_data.x = torch.tensor(x, dtype=torch.float)

    input_size = pyg_data.x.size(1)
    print(input_size)

    model = PYGModel(n_class, input_size, config['conv_class'], hidden_size=config['hidden_size'], num_layers=config['num_layers'])
    y_pred, score = model.fit_predict(pyg_data, full=True)
    y_pred = y_pred.argmax(axis=1)
    score = accuracy_score(y_test, y_pred)
    return score


r = optimize_function(func, search_space, time_budget=200, num_gpus=2, num_cpus=20, cpu_per_trial=2, gpu_per_trial=0.3)
show_results(r)


# with Timer():
    # y_pred = model.train_predict(dataset.get_data(), time_budget=100, n_class=n_class, schema=schema)
# score = accuracy_score(y_test, y_pred)
# print(score)


# %%

from uxils.ml.automl import show_glance

import xgboost as xgb
import lightgbm as lgb

# model = xgb.sklearn.XGBClassifier(n_jobs=20, gpu_id=0, colsample_bytree=0.9, subsample=0.9, n_estimators=50)
# model = lgb.sklearn.LGBMClassifier(n_jobs=20, device='gpu', n_estimators=200)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(accuracy_score(y_test, y_pred))

show_glance(X_train, y_train, X_test, y_test, accuracy_score, 15, method_name='predict')
