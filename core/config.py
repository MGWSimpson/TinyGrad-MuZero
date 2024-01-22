class BaseMuZeroConfig(object):
    def __init__(self):
        self.training_steps = None
        self.max_moves = None
        self.root_dirichlet_alpha = None
        self.root_exploration_fraction = None
        pass

    def get_uniform_network(self):
        raise NotImplementedError

    def new_game(self):
        raise NotImplementedError

    def visit_softmax_temperature_fn(self):
        raise NotImplementedError