import re
from array import array

import numpy
import psycopg2
import time
import numpy as np
import torch
conn = psycopg2.connect(database="postgres", user="postgres", host="114.115.156.203", port="2333")
cursor = conn.cursor()

'''a = torch.full((1,1),3.0,requires_grad=True)
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
print(b.grad)'''
string = '{0,0,1,2,3}'
str = string.replace("{", "[").replace("}", "]")
new_string = eval(str)
print(type(str))
print(new_string)
