import math

import numpy
import torch
from torch.optim import optimizer
from gdbc import GDBC
from sklearn import metrics as sk_metrics

'''a = torch.ones([3, 3],requires_grad=True)
b = torch.zeros([3, 3],requires_grad=True)
z = torch.exp(-2 * torch.sqrt(torch.sum(torch.pow(a-b, 1), 0)))
z.backward(torch.ones_like(z))
print(a.grad)
# e = torch.sqrt(torch.sum(torch.pow(a-b, 1), 0))
e = torch.tensor([1.7321, 1.7321, 1.7321], requires_grad=True)
print(e)
f = torch.exp(-2 * e)
f.backward(torch.ones_like(f))
print(e.grad)'''
x = torch.rand([10, 4],requires_grad=True)
y = torch.ones([10, 1],requires_grad=True)
w = torch.rand([4, 2],requires_grad=True)
p = torch.ones([10, 1],requires_grad=True)
# p = torch.matmul(x,w)
'''print(torch.matmul(x,w))
ridge = 0.01
hx = 1 / (1 + torch.pow(math.e, torch.matmul(x,w)))
loss = ridge*torch.mean(torch.pow(w,2))-torch.mean(y * torch.log(hx) + (1 - y) * torch.log(1 - hx))
loss.backward()
print(w.grad)'''
# print(x)
# print(w)
# p = torch.matmul(x,w)
'''a = torch.tensor([3.,4.,3], requires_grad=True)
b = torch.tensor([2.,2.,2], requires_grad=True)
q = torch.mul(a, b)
q.backward(torch.ones_like(q))
print(q)
print(a.grad)
print(b.grad)'''
# print(w.grad)
a = [1.,0.,1.,1.,1.]
b = [0.,0.,0.,0.,0.]
print(sk_metrics.roc_auc_score(a, b))



