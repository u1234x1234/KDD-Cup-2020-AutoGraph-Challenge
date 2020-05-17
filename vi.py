# %%
from uxils.serialization import load
import numpy as np
import sys
from uxils.ray_ext.utils import show_results

name = sys.argv[1]

data = load(f'task_{name}.pkl')

show_results(data, plot_results=False)
