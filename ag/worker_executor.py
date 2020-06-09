import atexit
import copy
import uuid

import cloudpickle
import ray
import zstandard
# from uxils.timer import Timer
# from uxils.profiling import Profiler

TASK_QUEUE_KEY = 'UXILS_TASK_QUEUE'
RESULTS_QUEUE_KEY = 'UXILS_RESULTS_QUEUE'
DATA_KEY = 'UXILS_DATA'


def serialize(value):
    return zstandard.ZstdCompressor().compress(cloudpickle.dumps(value))


def deserialize(value):
    return cloudpickle.loads(zstandard.ZstdDecompressor().decompress(value))


def worker(name, verbose=1):
    import torch
    print('CUDA', torch.cuda.is_available())

    client = ray.worker.global_worker.redis_client
    data = client.get(DATA_KEY)
    if verbose:
        print(f'Worker "{name}" data: {len(data)}')

    gdata = deserialize(data)
    while True:
        task = client.blpop(TASK_QUEUE_KEY)[1]
        task = deserialize(task)
        data = copy.deepcopy(gdata)

        if task is None:
            break

        name, task = task
        result = task(*data)

        client.rpush(RESULTS_QUEUE_KEY, serialize((name, result)))


class Executor:
    def __init__(self, n_workers, *args, gpu_per_trial):
        self._workers = []
        self._r_client = ray.worker.global_worker.redis_client
        self._r_client.set(DATA_KEY, serialize(args))

        for worker_id in range(n_workers):
            w_id = ray.remote(num_cpus=1, num_gpus=gpu_per_trial)(worker).remote(f'worker_{worker_id}')
            self._workers.append(w_id)
        atexit.register(self.stop)

    def apply(self, func, name=None):
        if name is None:
            name = str(uuid.uuid4())

        fs = serialize((name, func))
        self._r_client.rpush(TASK_QUEUE_KEY, fs)

    def get(self, timeout=None):
        r = self._r_client.blpop(RESULTS_QUEUE_KEY, timeout=timeout)
        if r is not None:
            return deserialize(r[1])

    def stop(self, force=True):
        for _ in range(len(self._workers)):
            self._r_client.lpush(TASK_QUEUE_KEY, serialize(None))

        if not force:
            raise NotImplementedError

        for w_id in self._workers:
            ray.cancel(w_id, force=True)
        for key in [TASK_QUEUE_KEY, RESULTS_QUEUE_KEY, DATA_KEY]:
            self._r_client.delete(key)
