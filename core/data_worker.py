
from mcts import MCTS, Node
class DataWorker(object):
    def __init__(self, id, config, shared_storage, replay_buffer):
        self.id = id
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer

    def run(self):
        #TODO: implement the data runner
        pass
