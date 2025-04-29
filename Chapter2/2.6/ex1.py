import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
time=100000
result=multinomial.Multinomial(time, fair_probs).sample()
print(result)