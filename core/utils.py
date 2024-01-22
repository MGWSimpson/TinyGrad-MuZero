import tinygrad
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.dtype import dtypes

from core.model import BaseMuZeroNet
from typing import List, Callable


#TODO: should be written into the Tensor class of TinyGrad
def scatter(self :Tensor,  src:Tensor, dim :int =0) -> Tensor:
    pass