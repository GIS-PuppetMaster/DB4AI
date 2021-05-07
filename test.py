import torch
from sklearn import metrics
from time import time


def gen_data(size, feature_num):
    x = torch.rand(size, feature_num)
    y = torch.zeros(size, 1)
    arg = torch.argmax(x, 1)
    for i in range(size):
        if arg[i, ...] > 1:
            y[i, ...] = 1
    return x, y


start_time = time()
feature_num = 4
iter_times = 10000
ridge = 0.01
learning_rate = 0.01

train_x, train_y = gen_data(10000, feature_num)
test_x, test_y = gen_data(1000, feature_num)

w = torch.rand(feature_num, 1, requires_grad=True)

for i in range(iter_times):
    # w.requires_grad_(True)
    hx = 1 / (1 + torch.exp(torch.matmul(train_x, w)))
    loss = ridge * torch.mean(torch.pow(w, 2)) - torch.mean(train_y * torch.log(hx) + (1 - train_y) * torch.log(1 - hx))
    if w.grad is not None:
        w.grad.data.zero_()
    loss.backward()
    g = w.grad
    w.data = w - learning_rate * g
w = w.detach()
pred = 1 / (1 + torch.exp(torch.matmul(test_x, w)))
for i in range(1000):
    if pred[i, ...] >= 0.5:
        pred[i, ...] = 1
    else:
        pred[i, ...] = 0
print(time() - start_time)
print(metrics.accuracy_score(pred, test_y))
