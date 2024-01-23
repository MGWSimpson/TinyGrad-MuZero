from data_worker import DataWorker
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
def _train():
    pass


def train(config,):
    storage = SharedStorage(config.get_uniform_network())
    replay_buffer = ReplayBuffer(batch_size=config.batch_size, capacity=config.window_size)

    workers = [DataWorker(rank=rank, config=config, shared_storage=storage, replay_buffer=replay_buffer)
               for rank in range(config.num_actors)]

    for worker in workers:
        worker.run()


