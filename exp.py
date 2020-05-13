# %%
from data_utils import read_dataset
from ag.model import Model
from sklearn.metrics import accuracy_score
from uxils.timer import Timer
from ag.pyg_utils import generate_pyg_data
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


dataset, y_test = read_dataset('a')
n_class = dataset.get_metadata()['n_class']
schema = dataset.get_metadata()['schema']
time_budget = dataset.get_metadata()['time_budget']

# d = generate_pyg_data(dataset.get_data())
m = Model()
y_pred = m.train_predict(dataset.get_data(), time_budget=100, n_class=n_class, schema=schema)
# score = accuracy_score(y_test, y_pred)
print(y_pred.shape)

qwe
# %%



# pyg_data = generate_pyg_data(dataset.get_data())


from torch_geometric.nn import GatedGraphConv, GATConv, TAGConv
from uxils.automl.parametric_family import ParametricFamilyModel
from uxils.ray_ext import optimize_function, show_results
import numpy as np


search_space = {
    'conv_class': [
        # GCNConv,
        # partialclass(GCNConv, improved=True),
        # partialclass(GCNConv, normalize=True),

        partialclass(ChebConv, K=3),
        partialclass(ChebConv, K=7),

        # SAGEConv,
        # partialclass(SAGEConv, concat=True),
        # partialclass(SAGEConv, normalize=True),

        # partialclass(SplineConv, dim=3, kernel_size=5),
        # partialclass(GraphConv, aggr='mean'),
        # PCLS(GravNetConv, space_dimensions=16, propagate_dimensions=16, k=5),
        # partialclass(GatedGraphConv, aggr='mean'), partialclass(GatedGraphConv, aggr='add'),
        # TAGConv
    ],
    'num_layers': [2, 3],
    'hidden_size': [32, 64, 128],
}


def func(config):
    dataset, y_test = read_dataset('c')

    # print(dataset.fea_table.values.shape)
    # pca = TruncatedSVD(n_components=550, algorithm='arpack')
    # X = pca.fit_transform(dataset.fea_table.values[:, 1:])
    # dataset.fea_table = pd.DataFrame(X)
    # dataset.fea_table['node_index'] = pd.Series(np.random.randint(10, size=len(dataset.fea_table)))

    n_class = dataset.get_metadata()['n_class']
    schema = dataset.get_metadata()['schema']
    time_budget = dataset.get_metadata()['time_budget']

    model = Model(config['conv_class'], hidden_size=config['hidden_size'], num_layers=config['num_layers'])
    y_pred = model.train_predict(dataset.get_data(), time_budget=100, n_class=n_class, schema=schema)
    score = accuracy_score(y_test, y_pred)
    return score


r = optimize_function(func, search_space, time_budget=200, num_gpus=2, num_cpus=20, cpu_per_trial=2, gpu_per_trial=0.3)
show_results(r)


# with Timer():
    # y_pred = model.train_predict(dataset.get_data(), time_budget=100, n_class=n_class, schema=schema)
# score = accuracy_score(y_test, y_pred)
# print(score)


# X_train, y_train, X_test, y_test = read_dataset('a')


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
