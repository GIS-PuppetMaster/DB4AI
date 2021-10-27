import numpy
import torch

a = torch.ones((1, 1), requires_grad=True)
b = torch.Tensor(numpy.array([[1.0, 2, 3], [4, 5, 6]]))
s = a * b
s.backward()
print(s)
