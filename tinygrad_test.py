from tinygrad import Tensor, nn
from core.config import DiscreteSupport

import math


class LinearNet:
  def __init__(self):
    self.l1 = Tensor.kaiming_uniform(784, 128)
    self.l2 = Tensor.kaiming_uniform(128, 10)
  def __call__(self, x:Tensor) -> Tensor:
    return x.flatten(1).dot(self.l1).relu().dot(self.l2)

""" 
Reference : Appendix F => Network Architecture
& Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
Function only works presuming that numbers are between -1 and 1
"""
def scalar_transform(x):     
    assert (x >= -1 or x <=1).numpy().all()

    epsilon = 0.001
    sign = Tensor.floor(x) + Tensor.ceil(x)
    
    output = sign * (Tensor.sqrt(Tensor.abs(x) + 1) - 1 + epsilon * x)
    return output



"""

Again, function only works presuming that numbers are between -1 and 1
"""
def inverse_scalar_transform(logits, scalar_support):
    value_probs = Tensor.softmax(logits, axis=1)

    value_support = Tensor.ones(value_probs.shape)
    value_support = Tensor([[x for x in scalar_support.range]])
    
    value = (value_support * value_probs).sum(1, keepdim=True)
    
    assert (value >= -1 or value <=1).numpy().all()

    epsilon = 0.001
    sign = Tensor.floor(value) + Tensor.ceil(value)
    
    output = (((Tensor.sqrt(1 + 4 * epsilon * (Tensor.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
    output = sign * output

    
    return output


""" 
Clamp function, should be included in Tensor class ?
"""
def clamp(x: Tensor, min: int, max:int,) -> Tensor:
    min_tensor = Tensor.full(x.shape, min)
    max_tensor = Tensor.full(x.shape, max)
    return Tensor.minimum(Tensor.maximum(x, min_tensor), max_tensor)
 


"""
Not a general function, just serves my purpose
"""
def my_scatter(x: Tensor, pos: int, set_size: int):
    assert pos < set_size
    x = x.unsqueeze(-1)
    left_zeros= Tensor.zeros(x.shape[0], x.shape[1], pos)
    right_zeros = Tensor.zeros(x.shape[0], x.shape[1], set_size- pos -1)
    return left_zeros.cat(x, dim=2).cat(right_zeros, dim=2)

"""
My Phi
"""
def _phi(x, min, max, set_size: int):
    x = clamp(x, min= min, max=max)
    x_low = x.floor()
    x_high = x.ceil()
    p_high = (x - x_low)
    p_low = 1 - p_high
    x_high_idx, x_low_idx = math.floor(((x_high - min)[0][0]).item()), math.floor(((x_low - min)[0][0]).item())
    return my_scatter(p_high ,x_high_idx, set_size) + my_scatter(p_low, x_low_idx, set_size)


    




def soft_update(target, source, tau):
    target_state_dict = nn.state.get_state_dict(target)
    source_state_dict = nn.state.get_state_dict(source)
    for layer in target_state_dict:
        target_state_dict[layer] = target_state_dict[layer] * (1.0 - tau) + source_state_dict[layer] * tau
    
    nn.state.load_state_dict(target, state_dict=target_state_dict)







def main():
    source  = LinearNet()
    target = LinearNet()

    tau=1e-2

    soft_update(target, source, tau)
    pass



if __name__ == "__main__": 
    main()




