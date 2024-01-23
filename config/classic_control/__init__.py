from core.config import BaseMuZeroConfig, DiscreteSupport
from .model import MuZeroNet
import gym
from .env_wrapper import ClassicControlWrapper
class ClassicControlConfig(BaseMuZeroConfig):



    def __init__(self, args):
        super(ClassicControlConfig, self).__init__(
            training_steps=20000,
            test_interval=100,
            test_episodes=5,
            checkpoint_interval=20,
            max_moves=1000,
            discount=0.997,
            dirichlet_alpha=0.25,
            num_simulations=50,
            batch_size=128,
            td_steps=5,
            lr_init=0.05,
            lr_decay_rate=0.01,
            lr_decay_steps=10000,
            window_size=1000,
            value_loss_coeff=1,
            value_support=DiscreteSupport(-20, 20),
            reward_support=DiscreteSupport(-5, 5))
        self._set_user_args(args)
        self._set_game() # declares



    def _set_user_args(self, args):
        self.env_name = args.env
        self.num_actors = args.num_actors
        self.seed = args.seed
        pass

    def _set_game(self):
        game = self.new_game()
        self.obs_shape = game.reset().shape[0]
        self.action_space_size = game.action_space_size


    def get_uniform_network(self):
        return MuZeroNet(self.obs_shape, self.action_space_size,  self.reward_support.size, self.value_support.size)

    def new_game(self):
        env = gym.make(self.env_name)
        return ClassicControlWrapper(env, discount=self.discount, k=4)




