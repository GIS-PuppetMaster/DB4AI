import torch

a = torch.tensor([2.,], requires_grad=True)
b = torch.tensor([3.,], requires_grad=True)

for i in range(2):
    a = a * b
del b
a.backward()
b = b+1
print(b.grad)