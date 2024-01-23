
from core.mcts import MCTS, Node
import tinygrad
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.dtype import dtypes

from core.utils import select_action

class DataWorker(object):
    def __init__(self, rank, config, shared_storage, replay_buffer):
        self.rank = rank
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer

    def run(self):
        model = self.config.get_uniform_network()
        while self.shared_storage.get_counter() < self.config.training_steps:
            model.set_weights(self.shared_storage.get_weights()) # TODO: implement

            env = self.config.new_game()

            obs = env.reset(seed=self.config.seed + self.rank)

            done = False
            eps_reward, eps_steps, visit_entropies = 0, 0, 0
            trained_steps = self.shared_storage.get_counter()
            #_temperature = self.config.visit_softmax_temperature_fn(num_moves=len(env.history),
                        #                                            trained_steps=trained_steps)

            while not done and eps_steps <= self.config.max_moves:
                obs = Tensor(obs, dtype=dtypes.float32).unsqueeze(0)
                network_output = model.initial_inference(obs)

                action = eps_steps % 2
                obs, reward, done, info = env.step(action)

                eps_reward += reward
                eps_steps += 1

                """root = Node(0)
                obs = Tensor(obs, dtype=dtypes.float32).unsqueeze(0)
                network_output = model.initial_inference(obs)
                root.expand(env.to_play(), env.legal_actions(), network_output)
                root.add_exploration_noise(dirichlet_alpha=self.config.root_dirichlet_alpha,
                                           exploration_fraction=self.config.root_exploration_fraction)
                MCTS(self.config).run(root, env.action_history(), model)
                action, visit_entropy = select_action(root, temperature=_temperature, deterministic=False)
                obs, reward, done, info = env.step(action.index)
                env.store_search_stats(root)

                eps_reward += reward
                eps_steps += 1
                visit_entropies += visit_entropy"""


            env.close()
            self.replay_buffer.save_game(env)
            print(self.replay_buffer.size())