
class SharedStorage(object):
    def __init__(self, model):
        self.model = model

        # logging info previously below but has been cut for now
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)
