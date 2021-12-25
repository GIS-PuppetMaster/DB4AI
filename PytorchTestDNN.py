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
    t = time()
    cursor.execute(f"select * from {table_name}")
    data = cursor.fetch()
    print(table_name,time()-t)
    for i in range(len(data)):
        data[i] = list(data[i])
        if isinstance(data[i][0], int):
            data[i][0] = float(data[i][0])
    return torch.tensor(data, requires_grad=True)


total = 0
total_acc = 0
iter_time = 1
transfer_time = 0
for i in range(iter_time):
    s = time()
    t1 = time()
    gdbc = GDBC()
    gdbc.connect()
    x = table_to_tensor(gdbc, "real_puf_x")
    y = table_to_tensor(gdbc, "real_puf_y")
    test_x = table_to_tensor(gdbc, "real_test_puf_x")
    test_y = table_to_tensor(gdbc, "real_test_puf_y")
    t2 = time()
    transfer_time += t2*1e6-t1*1e6
    print("transfer_time:"+str(t2*1e6-t1*1e6))
    w_0 = torch.rand((64,10),requires_grad=True)
    w_0 = w_0 / 2
    w_0 = torch.tensor(w_0.tolist(),requires_grad=True)
    w_1 = torch.rand((10,4),requires_grad=True)
    w_1 = w_1 / 2
    w_1 = torch.tensor(w_1.tolist(),requires_grad=True)
    w_2 = torch.rand((4,1),requires_grad=True)
    w_2 = w_2 / 2
    w_2 = torch.tensor(w_2.tolist(),requires_grad=True)
    b_0 = torch.rand((1,10),requires_grad=True)
    b_1 = torch.rand((1,4),requires_grad=True)
    b_2 = torch.rand((1,1),requires_grad=True)
    m = torch.nn.LeakyReLU(0.1)
    for i in range(500):
        # print(torch.matmul(x,w_0)+b_0)
        output_0 = m(torch.matmul(x,w_0)+b_0)
        output_1 = m(torch.matmul(output_0,w_1)+b_1)
        output_2 = 1/(1 + torch.exp(-1 * (torch.matmul(output_1,w_2) + b_2)))
        loss = -torch.mean(y*torch.log(output_2+0.0001)+(1-y)*torch.log(1-output_2+0.0001))
        loss.backward()
        lr = 0.01
        w_0 = w_0 - lr * w_0.grad
        w_1 = w_1 - lr * w_1.grad
        w_2 = w_2 - lr * w_2.grad
        b_0 = b_0 - lr * b_0.grad
        b_1 = b_1 - lr * b_1.grad
        b_2 = b_2 - lr * b_2.grad
        w_0 = torch.tensor(w_0.tolist(),requires_grad=True)
        w_1 = torch.tensor(w_1.tolist(),requires_grad=True)
        w_2 = torch.tensor(w_2.tolist(),requires_grad=True)
        b_0 = torch.tensor(b_0.tolist(),requires_grad=True)
        b_1 = torch.tensor(b_1.tolist(),requires_grad=True)
        b_2 = torch.tensor(b_2.tolist(),requires_grad=True)

    output_0 = m(torch.matmul(test_x,w_0)+b_0)
    output_1 = m(torch.matmul(output_0,w_1)+b_1)
    output_2 = 1/(1 + torch.exp(-1 * (torch.matmul(output_1,w_2) + b_2)))
    pred = []
    for i in output_2:
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
print("avg_transfer_time:" + str(transfer_time / iter_time))
print("avg_time:" + str(total / iter_time))