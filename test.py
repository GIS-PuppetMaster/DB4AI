import torch

x = torch.ones((100, 4))
w = torch.randn((4, ))
print(torch.matmul(x, w).shape)
