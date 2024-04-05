import numpy as np
from scipy.stats import entropy
from tinygrad import Tensor, nn

import shutil
import os
import logging


def make_results_dir(exp_path, args):
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        shutil.rmtree(exp_path)
        os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    return exp_path, log_path

def init_logger(base_path):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)




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