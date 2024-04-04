import numpy as np
from scipy.stats import entropy
from tinygrad import Tensor, nn

def select_action(node, temperature, deterministic=True):
    visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i, _ in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        action_pos = np.argmax([v for v, _ in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return visit_counts[action_pos][1], count_entropy


def soft_update(target, source, tau):
    # TODO: Check new soft update is equivalent to the state dict method
    source_weights = source.get_weights()
    target_weights = target.get_weights() 

    
    for i in range(len(source_weights)):
        target_weights[i] = target_weights[i] * (1.0 - tau) + source_weights[i] * tau

    target.set_weights(target_weights)




  
def clamp(x: Tensor, min: int, max:int,) -> Tensor:
    min_tensor = Tensor.full(x.shape, min)
    max_tensor = Tensor.full(x.shape, max)
    return Tensor.minimum(Tensor.maximum(x, min_tensor), max_tensor)
    




def my_scatter(x: Tensor, pos: int, set_size: int):
    assert pos < set_size
    x = x.unsqueeze(-1)
    left_zeros= Tensor.zeros(x.shape[0], x.shape[1], pos)
    right_zeros = Tensor.zeros(x.shape[0], x.shape[1], set_size- pos -1)
    return left_zeros.cat(x, dim=2).cat(right_zeros, dim=2)