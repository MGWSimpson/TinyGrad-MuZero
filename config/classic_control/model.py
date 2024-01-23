import tinygrad
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.dtype import dtypes

from core.model import BaseMuZeroNet
from typing import List, Callable


class TinyNet:
    def __init__(self, layers):
        self.layers: List[Callable[[Tensor], Tensor]] = layers

    def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)


class MuZeroNet(BaseMuZeroNet):
    def __init__(self, input_size, action_space_n, reward_support_size, value_support_size):
        super(MuZeroNet, self).__init__()
        self.hx_size = 32
        self._representation = TinyNet([nn.Linear(input_size, self.hx_size),
                                Tensor.tanh])

        self._dynamics_state = TinyNet([nn.Linear(self.hx_size + action_space_n, 64),
                                Tensor.tanh,
                                nn.Linear(64, self.hx_size),
                                Tensor.tanh])

        self._dynamics_reward = TinyNet([nn.Linear(self.hx_size + action_space_n, 64),
                                 Tensor.leakyrelu,
                                 nn.Linear(64, reward_support_size)])

        self._prediction_actor = TinyNet([nn.Linear(self.hx_size, 64),
                                  Tensor.leakyrelu,
                                  nn.Linear(64, action_space_n)])

        self._prediction_value = TinyNet([nn.Linear(self.hx_size, 64),
                                  Tensor.leakyrelu,
                                  nn.Linear(64, value_support_size)])

        self.action_space_n = action_space_n

        """
        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)
        """

    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        return self._representation(obs_history)

    def dynamics(self, state, action):
        assert len(state.shape) == 2
        assert action.shape[1] == 1


        action_one_hot = Tensor.zeros((action.shape[0], self.action_space_n),  dtype=dtypes.float32)

        #TODO: implement scatter

        x = state.cat(action_one_hot, dim=1)
        next_state = self._dynamics_state(x)
        reward = self._dynamics_reward(x)
        return next_state, reward


