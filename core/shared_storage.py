import ray


@ray.remote
class SharedStorage(object):
    def __init__(self, weights):
        self.weights = weights
        self.step_counter = 0


    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
