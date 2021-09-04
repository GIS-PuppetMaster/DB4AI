import numpy as np
import torch
'''print(np.shape(0))
s = np.shape(0)
print(type(s))
print(len(s))
x = torch.randn(2, 3, requires_grad=True)
print(x)
s = x.mean()
print(s)
s.backward()
print(x.grad)
# x = torch.tensor([[1., 2., 3.][1., 2., 3.]], requires_grad=True)
x = torch.randn(4, 5, requires_grad=True)
print(x)
y = 2 * x
s = torch.sum(y, 0)
print(s)
s.backward(torch.ones_like(s))
print(x.grad)
a = torch.ones((3, 3), requires_grad=True)
b = torch.zeros((3, 3), requires_grad=True)
c = torch.pow(a-b, 2)
d = torch.sum(torch.pow(a-b, 2), 1)
e = torch.sqrt(torch.sum(torch.pow(a-b, 2), 1))
f = 2 * e
g = 0 - f
h = torch.exp(g)
d.backward(torch.ones_like(d))
print(a.grad)

c = torch.exp(0-2*torch.sqrt(torch.sum(torch.pow(a-b, 2), 1)))
print(c)
c.backward(torch.ones_like(c))
t = 2 * a
print(t)
s = torch.zeros((3, 3), requires_grad=True)
e = torch.exp(s)
e.backward(torch.ones_like(e))
print(a.grad)
print(b.grad)
print(s.grad)'''
a = torch.ones((3, 3), requires_grad=True)
b = torch.pow(a, 2)
c = torch.matmul(a, b)
c.backward(torch.ones_like(c))
print(c)
print(a.grad)

