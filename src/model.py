"""the simple baseline for autograph"""
import time

import numpy as np
from ag.system_ext import suppres_all_output
from env_utils import prepare_env


N_TOP = 3


class Model:
    def __init__(self):
        prepare_env()

        import ray
        ray.init(
            num_gpus=1, num_cpus=4, memory=1e10, object_store_memory=1e10,
            configure_logging=False, ignore_reinit_error=True,
            log_to_driver=False,
            include_webui=False
        )

    def train_predict(self, data, time_budget, n_class, schema):
        start_time = time.time()
        import torch
        from ag.worker_executor import Executor
        from ag.pyg_model import SEARCH_SPACE_FLAT, PYGModel, create_factory_method
        from ag.pyg_utils import generate_pyg_data

        data = generate_pyg_data(data)
        print('DATAINFO', data, time_budget, n_class)

        base_class = create_factory_method(n_classes=n_class)
        n_edge = data.edge_index.shape[1]
        executor = Executor(4 if n_edge < 400000 else 1, base_class, data, gpu_per_trial=0.25)
        print('CONFIG', len(SEARCH_SPACE_FLAT))

        for config in SEARCH_SPACE_FLAT:

            def func(base_class, data):
                model = base_class(**config)
                y_pred, score = model.fit_predict(data)
                return y_pred, score

            executor.apply(func, name={str(k): str(v) for k, v in config.items()})

        results = []
        while (len(results) != len(SEARCH_SPACE_FLAT)) and ((time.time() - start_time) < (time_budget - 4)):
            r = executor.get(timeout=2)
            if r is not None:
                results.append(r)
                sresults = list(sorted(results, key=lambda x: -x[1][1]))

        print('\n'.join([f'{r[0]} {r[1][1]}' for r in sresults]))
        predictions = np.array([r[1][0] for r in sresults[:N_TOP] if r[1][1] > sresults[0][1][1] - 0.02])
        print(predictions.shape)

        from scipy.stats import gmean
        from scipy.special import softmax
        # predictions = predictions.mean(axis=0)
        predictions = np.mean(softmax(predictions, axis=2), axis=0)

        return predictions.argmax(axis=1)

    def __del__(self):
        try:
            with suppres_all_output():
                import ray
                ray.shutdown()
                time.sleep(1)
        except:
            pass
