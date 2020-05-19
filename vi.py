# %%
from uxils.serialization import load
import numpy as np
import sys
from uxils.ray_ext.utils import show_results
import glob

name = sys.argv[1]
data = []
for path in glob.glob(f'task_{name}*pkl'):
    data += load(path)
    print(path, len(data))

show_results(data, plot_results=False)
