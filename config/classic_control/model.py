import tinygrad
import math
import weakref
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.dtype import dtypes

from core.model import BaseMuZeroNet
from typing import List, Callable

import numpy as np



class TinyNet:
    def __init__(self, layers):
        self.layers: List[Callable[[Tensor], Tensor]] = layers

    def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)





class MuZeroNet(BaseMuZeroNet):
    def __init__(self, input_size, action_space_n, reward_support_size, value_support_size, inverse_value_transform=None, inverse_reward_transform=None):
        super(MuZeroNet, self).__init__(inverse_value_transform=inverse_value_transform, inverse_reward_transform=inverse_reward_transform)
        self.hx_size = 32
        self._representation = TinyNet([nn.Linear(input_size, self.hx_size),
                                Tensor.tanh])

        self._dynamics_state = TinyNet([nn.Linear((self.hx_size + action_space_n), 64),
                                Tensor.tanh,
                                nn.Linear(64, self.hx_size),
                                Tensor.tanh])

        self._dynamics_reward = TinyNet([nn.Linear((self.hx_size + action_space_n), 64),
                                 Tensor.leakyrelu,
                                 nn.Linear(64, reward_support_size)])

        self._prediction_actor = TinyNet([nn.Linear(self.hx_size, 64),
                                  Tensor.leakyrelu,
                                  nn.Linear(64, action_space_n)])

        self._prediction_value = TinyNet([nn.Linear(self.hx_size, 64),
                                  Tensor.leakyrelu,
                                  nn.Linear(64, value_support_size)])

        self.action_space_n = action_space_n


        # need method to iteratively access all layers of all networks
        self.networks = [self._representation, self._dynamics_state, self._dynamics_reward,
                         self._prediction_actor, self._prediction_value]
        
        """
        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)
        """


    """
    Takes in a Tensor
    returns tensor
    """ 
    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        return self._representation(obs_history)

    def dynamics(self, state: Tensor, action: Tensor):
       
        """assert len(state.shape) == 2
        assert action.shape[1] == 1"""
        
        action_one_hot = action.one_hot(self.action_space_n).squeeze(1)
        x =  state.cat(action_one_hot, dim=1)
        next_state = self._dynamics_state(x)
        reward = self._dynamics_reward(x)

        return next_state, reward


    """
    Weights returned as numpy array
    """
    def get_weights(self, in_np=True):
        weights = []
        for network in self.networks:
            for layer in network.layers:
                if isinstance(layer, nn.Linear):

                    if in_np:
                        weights.append(layer.weight.numpy())
                    else:
                        weights.append(layer.weight)
        return weights


    """
    Weights defined as numpy arrays
    """
    def set_weights(self, weights):
        weight_pointer = 0
        for network in self.networks:
            for layer in network.layers:
                if isinstance(layer, nn.Linear) and weight_pointer < len(weights):
                    layer.weight = Tensor(weights[weight_pointer])
                    weight_pointer +=1





