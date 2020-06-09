# %%
import sys
sys.path.append('/home/u1234x1234/autograph2020/src')
from data_utils import read_dataset

from uxils.profiling import Profiler
from uxils.ray_ext import optimize_function, show_results
from model import Model
from sklearn.metrics import accuracy_score
from uxils.timer import Timer
import numpy as np

name = sys.argv[1]
dataset, y_test = read_dataset(name)
n_class = dataset.get_metadata()['n_class']
schema = dataset.get_metadata()['schema']
time_budget = dataset.get_metadata()['time_budget']

scores = []
for i in range(10):
    m = Model()
    with Timer('Total time'):
        y_pred = m.train_predict(dataset.get_data(), time_budget=100, n_class=n_class, schema=schema)

    del m
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    print(scores)

print(np.mean(scores))