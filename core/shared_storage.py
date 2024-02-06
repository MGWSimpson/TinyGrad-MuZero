import ray


@ray.remote
class SharedStorage(object):
    def __init__(self, model):
        self.model = model
        self.step_counter = 0
        # logging info previously below but has been cut for now

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)
