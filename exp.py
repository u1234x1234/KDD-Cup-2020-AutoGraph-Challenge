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


dataset, y_test = read_dataset('e')
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
# X = dataset.get_data()['fea_table'].drop('node_index', axis=1).to_numpy()
X = pyg_data.x.numpy()
X_sh = compute_nn_features(X, pyg_data.edge_index.t().tolist())


import torch
from torch.utils.data import DataLoader
from torch_geometric.nn import Node2Vec
loader = DataLoader(torch.arange(pyg_data.num_nodes), batch_size=128, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(pyg_data.num_nodes, embedding_dim=64, walk_length=10,
                 context_size=5, walks_per_node=10)
model, data = model.to(device), pyg_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
def train():
    model.train()
    total_loss = 0
    for subset in loader:
        optimizer.zero_grad()
        loss = model.loss(data.edge_index, subset.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, 40):
    loss = train()
    # if epoch % 10 == 0:
    print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))

model.eval()
with torch.no_grad():
    X_emb = model(torch.arange(pyg_data.num_nodes, device=device)).cpu().numpy()


# %%
X = X_emb
# X = np.hstack([X_sh, X_emb])

X_train = X[dataset.get_data()['train_indices']]
X_test = X[dataset.get_data()['test_indices']]

y_train = dataset.get_data()['train_label'].label.to_numpy()
assert len(y_train) == len(X_train)
assert len(X_test) == len(y_test)
print(X_train.shape, X_test.shape)

# model = ExtraTreesClassifier(n_estimators=500, n_jobs=20)
model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
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
        # GCNConv,
        # partialclass(GCNConv, improved=True),
        # partialclass(GCNConv, normalize=True),

        # partialclass(ChebConv, K=3),
        # partialclass(ChebConv, K=7),

        # SAGEConv,
        # partialclass(SAGEConv, concat=True),
        # partialclass(SAGEConv, normalize=True),

        # partialclass(GraphConv, aggr='add'),
        # partialclass(GraphConv, aggr='mean'),

        # partialclass(GravNetConv, space_dimensions=4, propagate_dimensions=8, k=3),  # TODO
        # partialclass(GravNetConv, space_dimensions=4, propagate_dimensions=8, k=4),

        # partialclass(GatedGraphConv, aggr='mean'),  # SWAP params
        # partialclass(GatedGraphConv, aggr='add'),

        # partialclass(GATConv, concat=True),
        # partialclass(GATConv, dropout=0.5),

        # partialclass(TAGConv, K=3),
        # partialclass(TAGConv, K=5),
        # partialclass(ARMAConv, dropout=0, num_layers=2, num_stacks=3),
        # partialclass(ARMAConv, dropout=0, num_layers=3, num_stacks=2),
        # partialclass(ARMAConv, dropout=0.5, num_layers=3, num_stacks=3),
        # partialclass(ARMAConv, dropout=0.5),

        # partialclass(SGConv, K=1),
        # partialclass(SGConv, K=2),

        partialclass(FeaStConv, heads=1)
    ],
    'num_layers': [2],
    'hidden_size': [32, 64],
    'in_dropout': [0.5, 0.8],
    'out_dropout': [0.5, 0.8],
    'n_iter': [300, 500, 700, 1000],
}


def func(config):
    dataset, y_test = read_dataset('a')

    # print(dataset.fea_table.values.shape)
    # pca = TruncatedSVD(n_components=550, algorithm='arpack')
    # X = pca.fit_transform(dataset.fea_table.values[:, 1:])
    # dataset.fea_table = pd.DataFrame(X)
    # dataset.fea_table['node_index'] = pd.Series(np.random.randint(10, size=len(dataset.fea_table)))

    n_class = dataset.get_metadata()['n_class']
    schema = dataset.get_metadata()['schema']
    time_budget = dataset.get_metadata()['time_budget']
    pyg_data = generate_pyg_data(dataset.get_data())

    # import torch
    # x = compute_nn_features(pyg_data.x.numpy(), pyg_data.edge_index.t().tolist())
    # x = StandardScaler().fit_transform(x)
    # x = pyg_data.x.numpy()
    # from sklearn.decomposition import PCA
    # x = PCA(n_components=1200, svd_solver='arpack').fit_transform(x).copy()
    # pyg_data.x = torch.tensor(x, dtype=torch.float)

    input_size = pyg_data.x.size(1)
    print(input_size)

    model = PYGModel(
        n_class, input_size, config['conv_class'], hidden_size=config['hidden_size'], num_layers=config['num_layers'],
        in_dropout=config['in_dropout'], out_dropout=config['out_dropout'], n_iter=config['n_iter'])
    y_pred, score = model.fit_predict(pyg_data, full=True)
    y_pred = y_pred.argmax(axis=1)
    score = accuracy_score(y_test, y_pred)
    return score


r = optimize_function(func, search_space, time_budget=200, num_gpus=2, num_cpus=20, cpu_per_trial=2, gpu_per_trial=0.5)
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
