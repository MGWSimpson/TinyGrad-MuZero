from typing import List



# TODO: refactor this?
class Player(object):
    def __init__(self, id=1):
        self.id = id

    def __eq__(self, other):
        if not isinstance(other, Player):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id




class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index



class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]


    def to_play(self) -> Player:
        return Player()



class Game:

    def __init__(self, env, action_space_size: int, discount: float, config=None):
        self.env = env
        self.obs_history = []
        self.history = []
        self.rewards = []

        self.child_visits = []
        self.root_values = []

        self.action_space_size = action_space_size
        self.discount = discount
        self.config = config

        pass

    def legal_actions(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError()


    def obs(self, i):
        raise NotImplementedError

    def make_target(self):
        # TODO: implement re-analyze
        raise NotImplementedError

    def action_history(self, idx=None):
        if idx is None:
            return ActionHistory(self.history, self.action_space_size)
        else:
            return ActionHistory(self.history[:idx], self.action_space_size)

    def __len__(self):
        return len(self.rewards)

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

    def to_play(self) -> Player:
        return Player()

    def store_search_stats(self, root, idx: int = None):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        if idx is None:
            self.child_visits.append([root.children[a].visit_count / sum_visits if a in root.children else 0
                                      for a in action_space])
            self.root_values.append(root.value())
        else:
            self.child_visits[idx] = [root.children[a].visit_count / sum_visits if a in root.children else 0
                                      for a in action_space]
            self.root_values[idx] = root.value()
