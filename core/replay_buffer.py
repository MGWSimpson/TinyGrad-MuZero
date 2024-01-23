import numpy as np
from tinygrad import Tensor
from core.game import Game
class ReplayBuffer(object):

    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.game_look_up = []
        self.base_idx = 0

    def save_game(self, game, ):
        self.buffer.append(game)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(game))]


    """
    Uniform random sampling
    """
    def sample_batch(self, num_unroll_steps: int):
        obs_batch, action_batch, reward_batch, value_batch, policy_batch = [], [], [], [], []

        # Sample from game positions
        indices = np.random.choice(len(self.game_look_up), self.batch_size)


        for idx in indices:
            game_id, game_pos = self.game_look_up[idx] # get the game the position relates to
            game_id -= self.base_idx
            game = self.buffer[game_id]
            #TODO: run re-analyze
            #TODO extract info from games...
            pass


        obs_batch = Tensor(obs_batch)
        action_batch = Tensor(action_batch)
        reward_batch = Tensor(reward_batch)
        value_batch = Tensor(value_batch)
        policy_batch = Tensor(policy_batch)

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

