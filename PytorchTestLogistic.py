from time import time

import pyodbc
import torch
from torch import nn
from sklearn.metrics import accuracy_score


class GDBC(object):
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.ans = None

    def connect(self):
        self.conn = pyodbc.connect('DRIVER={GaussMPP};SERVER=127.0.0.1;DATABASE=omm;UID=omm;PWD=Gauss@123')
        self.cursor = self.conn.cursor()

    def close(self):
        self.cursor.commit()
        self.cursor.close()
        self.conn.close()
        print("Successfully closed!")
        return True

    def execute(self, sql: str):
        try:
            self.cursor.execute(sql)
            self.cursor.commit()
            self.ans = self.cursor.fetchall()
            # if 'if_tensor_exists' not in sql and 'qp4ai_select' not in sql:
            #     assert self.ans[0][0]==0
        except :
            raise Exception(f'error sql:{sql}')
            # self.ans = "no result"

    def fetch(self):
        return self.ans


def table_to_tensor(cursor,table_name):
    cursor.execute(f"select * from {table_name}")
    data = cursor.fetch()
    for i in range(len(data)):
        data[i] = list(data[i])
    return torch.tensor(data,requires_grad=True)

# gdbc = GDBC()
# gdbc.connect()
total = 0
total_acc = 0
iter_time = 100
for i in range(iter_time):
    gdbc = GDBC()
    gdbc.connect()
    w = torch.rand((30,1),requires_grad=True)
    s = time()
    x = table_to_tensor(gdbc, "real_x")
    y = table_to_tensor(gdbc, "real_y")
    test_x = table_to_tensor(gdbc, "real_test_x")
    test_y = table_to_tensor(gdbc, "real_test_y")
    for i in range(1000):
        hx = 1/(1 + torch.exp(-1 * torch.matmul(x,w)))
        loss = 0.01 * torch.mean(torch.pow(w,2))-torch.mean(y*torch.log(hx) + (1-y)*torch.log(1-hx))
        loss.backward()
        a = w.grad
        t = w - 0.01 * a
        w = torch.tensor(t.tolist(),requires_grad=True)
    hx = 1/(1 + torch.exp(-1 * torch.matmul(test_x,w)))
    list_y = test_y.tolist()
    pred = []
    for i in hx:
        if i[0] > 0.5:
            pred.append([1.0])
        else:
            pred.append([0.0])
    print("acc:" + str(accuracy_score(test_y.detach().numpy(),pred)))
    total_acc += accuracy_score(test_y.detach().numpy(),pred)
    t = time() - s
    print("time:" + str(t))
    total += t
print("avg_acc:" + str(total_acc / iter_time))
print("avg_time:" + str(total / iter_time))
# class LR(nn.Module):
#     def __init__(self):
#         super(LR,self).__init__()
#         self.fc=nn.Linear(30,2)
#     def forward(self,x):
#         out=self.fc(x)
#         out=torch.sigmoid(out)
#         return out
#
# net=LR()
# criterion=nn.CrossEntropyLoss() # 使用CrossEntropyLoss损失
# optm=torch.optim.Adam(net.parameters()) # Adam优化
# epochs=1000 # 训练1000次
#
#
# def test(x,y):
#     cnt = 0
#     sum = 0
#     x = x.tolist()
#     y = y.tolist()
#     for i in range(len(x)):
#         if x[i][0] > 0.5 and y[i][0] == 1.0:
#             cnt += 1
#             sum += 1
#         elif x[i][0] <= 0.5 and y[i][0] == 0.0:
#             cnt += 1
#             sum += 1
#         else:
#             sum += 1
#     return cnt/sum
#
# for i in range(epochs):
#     # 指定模型为训练模式，计算梯度
#     net.train()
#     # 输入值都需要转化成torch的Tensor
#     y_hat=net(x)
#     y = y.tolist()
#     for i in range(len(y)):
#         if y[i] == 1.0:
#             y[i] = [0.0,1,0]
#         else:
#             y[i] = [1.0,0,0]
#     y = torch.tensor(y)
#     loss=criterion(y_hat,y) # 计算损失
#     optm.zero_grad() # 前一步的损失清零
#     loss.backward() # 反向传播
#     optm.step() # 优化
#     if (i+1)%100 ==0 : # 这里我们每100次输出相关的信息
#         # 指定模型为计算模式
#         net.eval()
#         test_out=net(test_x)
#         # 使用我们的测试函数计算准确率
#         accu=test(test_out,test_y)
#         print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i+1,loss.item(),accu))
#
