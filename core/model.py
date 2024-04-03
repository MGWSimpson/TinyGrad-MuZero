import typing
from typing import Dict, List
from .game import Action
from tinygrad import Tensor

class NetworkOutput(typing.NamedTuple):
    value:float
    reward:float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class BaseMuZeroNet:
    def __init__(self, inverse_value_transform, inverse_reward_transform):
        super(BaseMuZeroNet, self).__init__()
        self.inverse_value_transform = inverse_value_transform
        self.inverse_reward_transform = inverse_reward_transform
        self.training = True

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, action):
        raise NotImplementedError

    def initial_inference(self, obs) -> NetworkOutput:
        state = self.representation(obs)
        actor_logit, value = self.prediction(state)

      
        if not self.training:
            value = self.inverse_value_transform(value)

        return NetworkOutput(value, 0, actor_logit, state)


    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        state, reward = self.dynamics(hidden_state, action)
        actor_logit, value = self.prediction(state)

        
        if not self.training:
            value = self.inverse_value_transform(value)
            reward = self.inverse_reward_transform(reward)

        return NetworkOutput(value, reward, actor_logit, state)

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False



