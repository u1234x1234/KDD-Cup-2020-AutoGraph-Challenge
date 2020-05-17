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

dataset, y_test = read_dataset('c')
n_class = dataset.get_metadata()['n_class']
schema = dataset.get_metadata()['schema']
time_budget = dataset.get_metadata()['time_budget']


m = Model()
with Timer('Total time'):
    y_pred = m.train_predict(dataset.get_data(), time_budget=100, n_class=n_class, schema=schema)

score = accuracy_score(y_test, y_pred)
print(score)
