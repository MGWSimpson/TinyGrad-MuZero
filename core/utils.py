import numpy as np
from scipy.stats import entropy
from tinygrad import Tensor

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
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def scalar_transform(x):
    """ Reference : Appendix F => Network Architecture
    & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
     """
    epsilon = 0.001
    sign = Tensor.ones(x.shape).float().to(x.device)
    sign[x < 0] = -1.0
    output = sign * (Tensor.sqrt(Tensor.abs(x) + 1) - 1 + epsilon * x)
    return output



def value_phi(self, x):
    return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

def reward_phi(self, x):
    return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)


def _phi(x, min, max, set_size: int):
    x.clamp_(min, max)
    x_low = x.floor()
    x_high = x.ceil()
    p_high = (x - x_low)
    p_low = 1 - p_high

    target = Tensor.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
    x_high_idx, x_low_idx = x_high - min, x_low - min
    target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
    target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
    return target

