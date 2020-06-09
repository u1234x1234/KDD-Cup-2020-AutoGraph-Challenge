# %%
import sys
from collections import defaultdict
from functools import partial

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, normalize

from ag.pyg_model import PYGModel
from ag.pyg_utils import generate_pyg_data
from data_utils import read_dataset
from torch_geometric.nn import (ARMAConv, ChebConv, FeaStConv, GATConv,
                                GatedGraphConv, GCNConv, GINConv, GraphConv,
                                GravNetConv, RGCNConv, SAGEConv, SGConv,
                                SplineConv, TAGConv)
from uxils.automl.parametric_family import ParametricFamilyModel
from uxils.functools_ext import partialclass
from uxils.ray_ext import optimize_function, show_results
from uxils.timer import Timer


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

        # partialclass(GatedGraphConv, aggr='mean'),  # SWAP params
        # partialclass(GatedGraphConv, aggr='add'),

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

        partialclass(FeaStConv, heads=1),
    ],
    'num_layers': [1, 2],
    'hidden_size': [32, 64, 96],
    'in_dropout': [0.5],
    'out_dropout': [0.5],
    'n_iter': [100, 300, 700],
    'weight_decay': [0, 1e-3],
    'learning_rate': [0.01, 0.001],
}


def func(config):
    dataset, y_test = read_dataset(config['ds'])
    n_class = dataset.get_metadata()['n_class']
    pyg_data = generate_pyg_data(dataset.get_data())

    # import torch
    # x = compute_nn_features(pyg_data.x.numpy(), pyg_data.edge_index.t().tolist())
    # x = StandardScaler().fit_transform(x)
    # x = pyg_data.x.numpy()
    # from sklearn.decomposition import PCA
    # x = PCA(n_components=1200, svd_solver='arpack').fit_transform(x).copy()
    # pyg_data.x = torch.tensor(x, dtype=torch.float)

    input_size = pyg_data.x.size(1)

    model = PYGModel(
        n_class, input_size, config['conv_class'], hidden_size=config['hidden_size'], num_layers=config['num_layers'],
        in_dropout=config['in_dropout'], out_dropout=config['out_dropout'], n_iter=config['n_iter'],
        weight_decay=config['weight_decay'], learning_rate=config['learning_rate'])
    y_pred, score = model.fit_predict(pyg_data, full=True)
    y_pred = y_pred.argmax(axis=1)
    score = accuracy_score(y_test, y_pred)
    return score

from uxils.serialization import dump
import uuid
import copy

while True:
    dataset = np.random.choice(['a', 'b', 'c', 'd', 'e'])
    if dataset in ['a', 'b', 'e']:
        gg = 0.25
    else:
        gg = 1

    print(dataset)
    ss = copy.deepcopy(search_space)
    ss['ds'] = [dataset]
    r = optimize_function(func, ss, time_budget=60*60, num_gpus=2, num_cpus=40, cpu_per_trial=2, gpu_per_trial=gg)
    name = str(uuid.uuid4())
    print(dataset, name)
    dump(r, f'task_{dataset}_{name}.pkl')
    # show_results(r)
