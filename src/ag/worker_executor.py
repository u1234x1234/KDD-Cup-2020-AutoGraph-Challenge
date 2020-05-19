import copy

import cloudpickle
import ray
import zstandard

TASK_QUEUE_NAME = 'UX_TASK_QUEUE'
RESULTS_QUEUE_NAME = 'UX_RESULTS_QUEUE'


def compress_zstd(value):
    return zstandard.ZstdCompressor().compress(value)


def decompress_zstd(value):
    return zstandard.ZstdDecompressor().decompress(value)


def serialize(value):
    return compress_zstd(cloudpickle.dumps(value))


def deserialize(value):
    return cloudpickle.loads(decompress_zstd(value))


def worker(name):
    print(f'Worker "{name}" started.')
    client = ray.worker.global_worker.redis_client
    base_class = client.get('BASE_CLASS')
    data = client.get('DATA')
    print(len(base_class), len(data))
    base_class = deserialize(base_class)
    data = deserialize(data)

    while True:
        config = client.blpop(TASK_QUEUE_NAME)[1]
        if config is None:
            break
        config = deserialize(config)

        model = base_class(**config)
        result = model.fit_predict(copy.deepcopy(data))

        result = serialize((result, config))
        client.rpush(RESULTS_QUEUE_NAME, result)


class Executor:
    def __init__(self, n_workers, base_class, data):
        self._workers = []
        self._r_client = ray.worker.global_worker.redis_client
        base_class = serialize(base_class)
        data = serialize(data)

        self._r_client.set('BASE_CLASS', base_class)
        self._r_client.set('DATA', data)

        for worker_id in range(n_workers):
            w_id = ray.remote(num_cpus=1, num_gpus=0.3)(worker).remote(f'worker_{worker_id}')
            self._workers.append(w_id)

    def apply(self, func):
        fs = serialize(func)
        self._r_client.rpush(TASK_QUEUE_NAME, fs)

    def get(self, timeout=None):
        r = self._r_client.blpop(RESULTS_QUEUE_NAME, timeout=timeout)
        if r is not None:
            return deserialize(r[1])

    def stop(self):
        for w_id in self._workers:
            ray.cancel(w_id, force=True)
