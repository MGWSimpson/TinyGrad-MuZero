"""
Parent class for implementing different configs for the different types of games
"""
from abc import abstractmethod

from tinygrad import Tensor

import numpy as np

from core.utils import clamp, my_scatter

import math

class DiscreteSupport:
    def __init__(self, min: int, max: int):
        assert min < max
        self.min = min
        self.max = max
        self.range = range(min, max + 1)
        self.size = len(self.range)


class BaseMuZeroConfig(object):
    def __init__(self,
                 training_steps: int,
                 test_interval: int,
                 test_episodes: int,
                 checkpoint_interval: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 lr_init: float,
                 lr_decay_rate: float,
                 lr_decay_steps: float,
                 window_size: int = int(1e6),
                 value_loss_coeff: float = 1,
                 value_support: DiscreteSupport = None,
                 reward_support: DiscreteSupport = None):
        # Self-Play
        self.action_space_size = None

        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount
        self.max_grad_norm = 5

        # testing arguments
        self.test_interval = test_interval
        self.test_episodes = test_episodes

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the environment, we can use them to
        # initialize the rescaling. This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.max_value_bound = None
        self.min_value_bound = None

        # Training
        self.training_steps = training_steps
        self.checkpoint_interval = checkpoint_interval
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll_steps = 3
        self.td_steps = td_steps
        self.value_loss_coeff = value_loss_coeff
        self.device = 'gpu'
        self.exp_path = './'  # experiment path
        self.debug = False
        self.model_path = 'model.safetensors'
        self.seed = None
        self.value_support = value_support
        self.reward_support = reward_support
        self.num_actors = None

        # optimization control
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

        # replay buffer
        self.priority_prob_alpha = 1
        self.use_target_model = True
        self.revisit_policy_search_rate = 0
        self.use_max_priority = None


    @abstractmethod
    def get_uniform_network(self):
        raise NotImplementedError

    @abstractmethod
    def new_game(self):
        raise NotImplementedError

    @abstractmethod
    def _set_user_args(self, args):
        raise NotImplementedError


    @abstractmethod
    def visit_softmax_temperature(self, num_moves, trained_steps):
        raise NotImplementedError

    def inverse_reward_transform(self, reward_logits):
        return self.inverse_scalar_transform(reward_logits, self.reward_support)

    def inverse_value_transform(self, value_logits):
        return self.inverse_scalar_transform(value_logits, self.value_support)

    """
    Takes in a Tensor, returns a Tensor
    """
    def inverse_scalar_transform(self, logits, scalar_support):
        value_probs = Tensor.softmax(logits, axis=1)

        value_support = Tensor.ones(value_probs.shape)
        value_support = Tensor([[x for x in scalar_support.range]])
        
        value = (value_support * value_probs).sum(1, keepdim=True)
        
        assert (value >= -1 or value <=1).numpy().all()

        epsilon = 0.001
        sign = Tensor.floor(value) + Tensor.ceil(value)
        
        output = (((Tensor.sqrt(1 + 4 * epsilon * (Tensor.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        output = sign * output

        
        return output

    
    """
    Takes in a tensor
    Returns a tensor
    """
    @staticmethod
    def scalar_transform(x):
        #TODO: unsure if assertion required, assert (x >= -1 or x <=1).numpy().all()

        epsilon = 0.001
        sign = Tensor.floor(x) + Tensor.ceil(x)
        
        output = sign * (Tensor.sqrt(Tensor.abs(x) + 1) - 1 + epsilon * x)
        return output


    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

  
    
    """
    X is a tensor
    Returns a tensor
    """
    @staticmethod
    def _phi(x, min, max, set_size: int):
        
        x = clamp(x, min= min, max=max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = (x - x_low)
        p_low = 1 - p_high
        x_high_idx, x_low_idx = math.floor(((x_high - min)[0][0]).item()), math.floor(((x_low - min)[0][0]).item())
        return my_scatter(p_high ,x_high_idx, set_size) + my_scatter(p_low, x_low_idx, set_size)

    

    """
    Takes in a Tensors
    Returns a Tensor
    """
    def scalar_reward_loss(self, prediction, target):
        assert type(prediction) == Tensor
        assert type(target) == Tensor

        return -(Tensor.log_softmax(prediction, axis=1) * target).sum(1)

    """
    Takes in a a tensor
    Returns a tensor
    """
    def scalar_value_loss(self, prediction, target):
        return -(Tensor.log_softmax(prediction, axis=1) * target).sum(1)
