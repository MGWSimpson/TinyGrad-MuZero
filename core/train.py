import pickle
from tinygrad import Tensor, nn
import ray
import time
from core.data_worker import DataWorker
from core.shared_storage import SharedStorage
from core.replay_buffer import ReplayBuffer
from core.utils import adjust_lr, soft_update

def _train(config, shared_storage, replay_buffer):

    model = config.get_uniform_network()
    target_model = config.get_uniform_network()

    optim = nn.optim.Adam(params=None, lr=config.lr_init,)

    while ray.get(replay_buffer.size.remote()) == 0:
        pass


    for step_count in range(config.training_steps):

        shared_storage.incr_counter.remote()

        lr = adjust_lr(config, optim, step_count)

        if step_count % config.checkpoint_interval == 0:  ## after a certain number of training steps, save the model so the workers can use it
            shared_storage.set_weights.remote(model.get_weights())

        update_weights(model, target_model, optim, replay_buffer, config)

        # softly update target model
        if config.use_target_model:
            soft_update(target_model, model, tau=1e-2)
            target_model.eval()

        if step_count % 50 == 0:
            replay_buffer.remove_to_fit.remote()


    shared_storage.set_weights.remote(model.get_weights())


def update_weights(model, target_model, optim, replay_buffer, config):
    # TODO: start from here
    pass

def train(config,):

    storage = SharedStorage.remote(config.get_uniform_network().get_weights())
    replay_buffer = ReplayBuffer.remote(batch_size=config.batch_size, capacity=config.window_size)
    workers = [DataWorker.remote(rank, config, storage, replay_buffer)
               for rank in range(0, config.num_actors)]
    worker_waitables = [worker.run.remote() for worker in workers]


    _train(config, storage, replay_buffer)

    ray.wait(ray_waitables=worker_waitables, num_returns=len(workers))
    print(ray.get(replay_buffer.size.remote()))
