import torch
import numpy as np
from time import time
s = time()
x = torch.randn((1000, 4))
y = torch.randint(low=0, high=2, size=(1000, 1))
w = torch.randn((4, 1), requires_grad=True)
lr = 0.01
threshold = 0.3
iter_times = 10000
i=0
while True:
    w.requires_grad = True
    hx = 1/(1+torch.pow(np.e, torch.matmul(x,w)))
    loss = torch.mean(y*torch.log(hx)+(1-y)*torch.log(1-hx))
    loss.backward()
    print(loss)
    g = w.grad
    with torch.no_grad():
        w = w + w.grad * lr
    i+=1
    if i>=iter_times:
        break
print(f'time:{time()-s} s')