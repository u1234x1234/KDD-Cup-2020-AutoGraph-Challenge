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
            num_gpus=1, num_cpus=5, memory=1e10, object_store_memory=1e10,
            configure_logging=False, ignore_reinit_error=True,
            log_to_driver=True,
            include_webui=False
        )

    def train_predict(self, data, time_budget, n_class, schema):
        start_time = time.time()
        from ag.worker_executor import Executor
        from ag.pyg_model import SEARCH_SPACE_FLAT, PYGModel, create_factory_method
        from ag.pyg_utils import generate_pyg_data
        from dgl import DGLGraph
        import torch

        print(data['fea_table'])
        print(data['fea_table'].shape)
        n_edge = len(data['edge_file'])
        print('n_edge', n_edge)
        n_cv = 1 if n_edge > 400000 else 2
        data = generate_pyg_data(data, n_cv)
        print('EDGE WEIGHTS', data.edge_weight, data.edge_weight.mean(), data.edge_weight.sum(), data.edge_weight.max(), data.edge_weight.shape)

        g = DGLGraph((data.edge_index[1], data.edge_index[0]))
        # g.add_edges(data.edge_index[1], data.edge_index[0])

        print('DATAINFO', data, time_budget, n_class)

        base_class = create_factory_method(n_classes=n_class)
        executor = Executor(5 if n_edge < 400000 else 1, base_class, data, g, gpu_per_trial=0.2)
        print('CONFIG', len(SEARCH_SPACE_FLAT))
        SEARCH_SPACE_FLAT = SEARCH_SPACE_FLAT * 2

        for config in SEARCH_SPACE_FLAT:

            def func(base_class, data, g):
                model = base_class(**config)
                y_pred, score = model.fit_predict(data, g)
                return y_pred, score

            executor.apply(func, name={str(k): str(v) for k, v in config.items()})

        results = []
        while (len(results) != len(SEARCH_SPACE_FLAT)) and ((time.time() - start_time) < (time_budget - 4)):
            r = executor.get(timeout=2)
            if r is not None:
                results.append(r)
                sresults = list(sorted(results, key=lambda x: -x[1][1]))
                print(len(results), time.time() - start_time)

        print('\n'.join([f'{r[0]} {r[1][1]}' for r in sresults]))
        predictions = np.vstack([r[1][0] for r in sresults[:N_TOP] if r[1][1] > sresults[0][1][1] - 0.02])
        print(predictions.shape)

        from scipy.stats import gmean
        predictions = gmean(predictions, axis=0)

        return predictions.argmax(axis=1)

    def __del__(self):
        try:
            with suppres_all_output():
                import ray
                ray.shutdown()
                time.sleep(1)
        except:
            pass
