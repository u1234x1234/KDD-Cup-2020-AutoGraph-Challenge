# %%
from uxils.serialization import load
import numpy as np
import sys
import json
import re
from uxils.ray_ext.utils import show_results
import glob

name = sys.argv[1]
data = []
for path in glob.glob(f'task_{name}*pkl'):
    data += load(path)
    print(path, len(data))
show_results(data, plot_results=False, n_top=40)


data = []
for path in glob.glob(f'dgl4_task_{name}*pkl'):
    data += load(path)
    print(path, len(data))
configs = []
for config, (scores, time) in data:
    # n_iters, scores = zip(*scores)
    config = re.findall(r'({.+})', config)[0]
    for n_iter, score in scores:
        configs.append((score, n_iter, time, config))

configs = list(sorted(configs, key=lambda x: -x[0]))
print('\n'.join([str(x) for x in configs[:40]]))
