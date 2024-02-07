import numpy as np
from scipy.stats import entropy


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



def adjust_lr(config, optimizer, step_count):
    lr = config.lr_init * config.lr_decay_rate ** (step_count / config.lr_decay_steps)
    lr = max(lr, 0.001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
