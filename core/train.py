import pickle

import ray
import time
from core.data_worker import DataWorker
from core.shared_storage import SharedStorage
from core.replay_buffer import ReplayBuffer
def _train():
    pass
def train(config,):

    storage = SharedStorage.remote(config.get_uniform_network().get_weights())
    replay_buffer = ReplayBuffer.remote(batch_size=config.batch_size, capacity=config.window_size)



    workers = [DataWorker.remote(rank, config, storage, replay_buffer)
               for rank in range(0, config.num_actors)]


    worker_waitables = [worker.run.remote() for worker in workers]


    ray.wait(ray_waitables=worker_waitables, num_returns=len(workers))
    print(ray.get(replay_buffer.size.remote()))
