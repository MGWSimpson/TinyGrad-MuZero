import typing
from typing import Dict, List
from .game import Action


class NetworkOutput(typing.NamedTuple):
    value:float
    reward:float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class BaseMuZeroNet:
    def __init__(self ):
        super(BaseMuZeroNet, self).__init__()

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, action):
        raise NotImplementedError

    def initial_inference(self, obs) -> NetworkOutput:
        state = self.representation(obs)
        actor_logit, value = self.prediction(state)
        return NetworkOutput(value, 0, actor_logit, state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        state, reward = self.dynamics(hidden_state, action)
        actor_logit, value = self.prediction(state)
        return NetworkOutput(value, reward, actor_logit, state)

    def get_weights(self):
        raise NotImplementedError # TODO

    def set_weights(self, weights):
        raise NotImplementedError # TODO
