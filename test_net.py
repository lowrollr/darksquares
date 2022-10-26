from net import BeliefNet
import torch

b = BeliefNet(22, 8)

input = torch.rand(size=(4,22,8,8))
expected = torch.rand(size=(4,6,8,8)), torch.rand(size=(4,8)), torch.rand(size=(4,2))



out = b(input)
for a in out:
    print(a.shape)

b.loss(input, out, expected)