"""
Parent class for implementing different configs for the different types of games
"""
from abc import abstractmethod

from tinygrad import Tensor

import numpy as np


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
        self.num_unroll_steps = 5
        self.td_steps = td_steps
        self.value_loss_coeff = value_loss_coeff
        self.device = 'gpu'
        self.exp_path = None  # experiment path
        self.debug = False
        self.model_path = None
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
    Takes in a numpy, returns a numpy
    """
    def inverse_scalar_transform(self, logits, scalar_support):
        """ Reference : Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """

        logits = Tensor(logits)

        value_probs = Tensor.softmax(logits, axis=1)
        value_support = Tensor.ones(value_probs.shape)
        
        value_support = Tensor([[x for x in scalar_support.range]])
        
        value_support = value_support.to(device=value_probs.device)

        
        value = (value_support * value_probs).sum(axis=1, keepdim=True)

        

        epsilon = 0.001
        sign = Tensor.ones(value.shape).float().to(value.device)
        

        #sign[value < 0] = -1.0 ?? 

        output = (((Tensor.sqrt(1 + 4 * epsilon * (Tensor.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        output = sign * output
        return output.numpy()

    
    """
    Takes in a tensor
    Returns a tensor
    """
    @staticmethod
    def scalar_transform(x):
        """ Reference : Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        epsilon = 0.001
        sign = Tensor.ones(x.shape).float().to(x.device)
        #sign[x < 0] = -1.0 ??
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
        

        x = x.numpy()

        x = np.clip(x, a_max=max, a_min=min)
        

        x_low = Tensor(np.floor(x ))
        x_high = Tensor(np.ceil(x))


        x = Tensor(x)

        p_high = (x - x_low)
        p_low = 1 - p_high

        

        target = Tensor.zeros(x.shape[0], x.shape[1], set_size).to(x.device)

        x_high_idx, x_low_idx = x_high - min, x_low - min
        

        

        #target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1)) TODO
        #target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target
    

    """
    Takes in a Tensors
    Returns a Tensor
    """
    def scalar_reward_loss(self, prediction, target):
        return -(Tensor.log_softmax(prediction, axis=1) * target).sum(1)

    """
    Takes in a a tensor
    Returns a tensor
    """
    def scalar_value_loss(self, prediction, target):
        return -(Tensor.log_softmax(prediction, axis=1) * target).sum(1)
