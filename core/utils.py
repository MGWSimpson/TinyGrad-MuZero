import tinygrad
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.dtype import dtypes

from core.model import BaseMuZeroNet
from typing import List, Callable


def select_action(root, temperature, deterministic):
    raise NotImplementedError




