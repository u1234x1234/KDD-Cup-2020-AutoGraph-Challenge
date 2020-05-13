"""the simple baseline for autograph"""
import time

import numpy as np
import pandas as pd

from .parametric_family import ParametricFamilyModel
from .pyg_model import SEARCH_SPACE, PYGModel
from .pyg_utils import generate_pyg_data


class Model:
    def __init__(self):
        self.p_model = ParametricFamilyModel(
            PYGModel, SEARCH_SPACE, parallel_predictions=False,
            cpu_per_trial=1, gpu_per_trial=0.5,
        )
        import ray
        ray.init(
            num_gpus=1, num_cpus=4, memory=10e9, object_store_memory=10e9,
            configure_logging=False, ignore_reinit_error=True,
            log_to_driver=True,
            include_webui=False
        )

    def train_predict(self, data, time_budget, n_class, schema):
        data = generate_pyg_data(data)
        import ray
        data_id = ray.put(data)
        self.p_model.start(data_id, 100)
        time.sleep(10)
        print('PREDICT')
        return self.p_model.predict(data_id)

    def __del__(self):
        try:
            if hasattr(self, 'p_model'):
                self.p_model.executor.stop()
        except Exception as e:
            print(e)
        try:
            import ray
            ray.shutdown()
        except:
            pass
