"""
Parent class for implementing different configs for the different types of games
"""
from abc import abstractmethod

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
        self.device = 'cpu'
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