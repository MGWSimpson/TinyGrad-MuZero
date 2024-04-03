import ray
import numpy as np
from tinygrad import Tensor
from core.game import Game
@ray.remote
class ReplayBuffer(object):

    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = [] # stores env_wrapper / game objects
        self.game_look_up = []
        self.base_idx = 0

    def save_game(self, env):
        self.buffer.append(env)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(env))]


    """
    Uniform random sampling
    """
    def sample_batch(self, num_unroll_steps: int,td_steps: int,  model=None, config=None ):
        obs_batch, action_batch, reward_batch, value_batch, policy_batch = [], [], [], [], []

        # Sample from game positions
        indices = np.random.choice(len(self.game_look_up), self.batch_size)


        for idx in indices:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]

            obs_history,history,rewards,action_space_size,discount = game
            

            _actions = history[game_pos:game_pos + num_unroll_steps]
            # random action selection to complete num_unroll_steps
            _actions += [np.random.randint(0, action_space_size)
                         for _ in range(num_unroll_steps - len(_actions))]

            obs_batch.append(game.obs(game_pos))
            action_batch.append(_actions)

            value, reward, policy = game.make_target(game_pos, num_unroll_steps, td_steps, model, config)
            reward_batch.append(reward)
            value_batch.append(value)
            policy_batch.append(policy)


        return obs_batch, action_batch, reward_batch, value_batch, policy_batch, indices

    def remove_to_fit(self):
        if self.size() > self.capacity:
            num_excess_games = self.size() - self.capacity
            excess_games_steps = sum([len(game) for game in self.buffer[:num_excess_games]])
            del self.buffer[:num_excess_games]
            del self.game_look_up[:excess_games_steps]
            self.base_idx += num_excess_games

    def size(self):
        return len(self.buffer)

