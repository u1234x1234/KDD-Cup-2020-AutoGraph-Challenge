"""Incapsulates the model parametrized by the config space.
"""
import os
from typing import List

import numpy as np
from scipy.special import softmax

from .ray_ext import (AsyncExecutor, AsyncHyperBandScheduler,
                      BasicVariantGenerator, _create_trainable, optimize_class,
                      ray_context)

SCORE_NAME = 'score'
STEP_NAME = 'training_iteration'


def _make_predictions(base_class, config, checkpoint_path, X, trainset=None, n_seconds=20):
    """Initialize the model and make predictions. Optionally refit a restored model on the new data.
    Although the predict&train stages are semantically different,
    they are fused due to efficiency considerations.
    """
    model = base_class(**config)
    model.restore(checkpoint_path)
    y_pred = model.predict(X)
    print(y_pred.shape)
    return y_pred


class ParametricFamilyModel:
    "Search for the best parameters given the CV and metric"
    def __init__(self, base_class, search_space: dict, parallel_predictions=False,
                 n_gpus=0, n_cpus=1, gpu_per_trial=0, cpu_per_trial=1):
        self._base_class = base_class
        self._search_space = search_space
        self._parallel_predictions = parallel_predictions

        self.n_gpus = n_gpus
        self.n_cpus = n_cpus
        self.gpu_per_trial = gpu_per_trial
        self.cpu_per_trial = cpu_per_trial
        self.executor = None

    def start(self, data_id, time_budget, seconds_per_step=10, max_t=5, reduction_factor=4):
        import ray

        def data_getter():  # TODO check speed up
            return ray.get(data_id)

        trainable = _create_trainable(self._base_class, None, data_getter, seconds_per_step=seconds_per_step)

        search_alg = BasicVariantGenerator(
            shuffle=True
        )

        scheduler = AsyncHyperBandScheduler(metric='score', max_t=max_t, reduction_factor=reduction_factor)

        resources_per_trial = {'cpu': self.cpu_per_trial, 'gpu': self.gpu_per_trial}
        self.executor = AsyncExecutor(
            trainable,
            config=self._search_space,
            num_samples=5,
            resources_per_trial=resources_per_trial,
            shuffle=True,
            scheduler=scheduler,
            search_alg=search_alg,
            checkpoint_at_end=False,
            checkpoint_freq=1,
        )

    def predict(self, data_id, n_top=3, refit_seconds=30):
        results, n_incompleted = self.executor.get_results()
        if len(results) == 0:
            return None

        results = list(sorted(results, key=lambda x: -x[SCORE_NAME]))

        to_predict = []
        for info in results[:n_top]:
            print(info)
            checkpoint_path = os.path.join(
                info['logdir'], f"checkpoint_{info[STEP_NAME]}/model")
            to_predict.append((info['config'], checkpoint_path))

        predictions = [_make_predictions(self._base_class, config, path, data_id, refit_seconds)
                       for config, path in to_predict]

        y_pred = np.stack(predictions).mean(axis=0)
        # y_pred = softmax(y_pred, axis=1)

        return y_pred
