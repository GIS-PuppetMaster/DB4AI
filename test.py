import re
from array import array

import numpy
import psycopg2
import time
import numpy as np
import torch
import Nodes as nd
import Nodes
from gdbc import GDBC
'''conn = psycopg2.connect(database="postgres", user="postgres", host="114.115.156.203", port="2333")
cursor = conn.cursor()
a = torch.full((1,1),3.0,requires_grad=True)
# a = torch.tensor([[1.0,2.0,3.],[4.,5.,6.]],requires_grad=True)
# a = torch.full((3,3),2.0,requires_grad=True)
# a = torch.tensor([[1.0,1.0,1.],[1.0,1.0,1.],[4.,5.,6.]],requires_grad=True)
print(a)
b = torch.tensor([[1.0,2.0,3.0],[3.0,4.0,5.0],[5.0,6.0,7.0]],requires_grad=True)
print(b)
s = torch.mul(a,b)
print(s)
s.backward(torch.ones_like(s))
print(a.grad)
print(b.grad)

a = torch.full((1,1),1.0,requires_grad=True)
b = torch.zeros((1,3),requires_grad=True)
a = torch.tensor([3.0,3.0,3.0],requires_grad=True)

b = torch.tensor([1.732, 1.732, 1.732],requires_grad=True)
print(b)
s = torch.exp(-2*b) # s = [0.0313, 0.0313, 0.0313]
print(s)
s.backward(torch.ones_like(s))
print(b.grad)
a = torch.full((3,3),1.0,requires_grad=True)
b = torch.full((3,3),0.0,requires_grad=True)
print(torch.sum(a,0))
c = torch.exp(-2*torch.sqrt(torch.sum(torch.pow((a-b),1),0)))
c.backward(torch.ones_like(c))
print(a.grad)
print(b.grad)'''
ss = nd.InstantiationClass(1, 'Add',
                           with_grad=True)
print('backward' in dir(ss))