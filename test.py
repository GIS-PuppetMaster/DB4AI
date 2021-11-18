import math

import numpy
import torch
import unittest
from torch.optim import optimizer
from gdbc import GDBC
from sklearn import metrics as sk_metrics
from Nodes import str_to_list

'''class Test:

    def test(self, cursor):
        cursor.execute("select data from x")
        a = str_to_list(cursor.fetch()[0][0])
        lx = []
        s = []
        j = 0
        for i in a:
            if j < 30:
                j += 1
                s.append(i)
            else:
                j = 1
                lx.append(s)
                s = [i]
        lx.append(s)
        x = torch.tensor(lx,requires_grad=True)
        print(x)
        cursor.execute("select data from y")
        a = str_to_list(cursor.fetch()[0][0])
        ly = []
        for i in a:
            ly.append([float(i)])
        y = torch.tensor(ly,requires_grad=True)
        cursor.execute("select data from __0w")
        a = str_to_list(cursor.fetch()[0][0])
        lw = []
        for i in a:
            lw.append([i])
        w = torch.tensor(lw,requires_grad=True)
        for i in range(3):
            hx = 1.0/(1 + torch.exp(-1 * torch.matmul(x,w)))
            loss = 0.0001 * torch.mean(torch.pow(w,2)) - torch.mean(y * torch.log(hx) + (1-y) * torch.log(1-hx))
            loss.backward()
            wgrad = w.grad
            w = w - w.grad * 0.01'''



