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
        if isinstance(data[i][0], int):
            data[i][0] = float(data[i][0])
    return torch.tensor(data, requires_grad=True)


total = 0
total_acc = 0
iter_time = 100
for i in range(iter_time):
    s = time()
    gdbc = GDBC()
    gdbc.connect()
    x = table_to_tensor(gdbc, "real_multi_class_x")
    y = table_to_tensor(gdbc, "real_multi_class_y")
    test_x = table_to_tensor(gdbc, "real_multi_class_test_x")
    test_y = table_to_tensor(gdbc, "real_multi_class_test_y")
    w = torch.rand((6,4),requires_grad=True)
    b = torch.rand((1,4),requires_grad=True)
    for i in range(2000):
        hx = torch.softmax(torch.matmul(x,w)+b,1)
        loss = -torch.mean(y*torch.log(hx))
        loss.backward()
        a = w.grad
        t = w - 0.01 * a
        w = torch.tensor(t.tolist(),requires_grad=True)
    hx = torch.softmax(torch.matmul(test_x,w)+b,1)
    test_y = torch.argmax(test_y, 1)
    pred = torch.argmax(hx, 1)
    print("acc:" + str(accuracy_score(test_y.detach().numpy(),pred.detach().numpy())))
    total_acc += accuracy_score(test_y.detach().numpy(),pred.detach().numpy())
    t = time() - s
    print("time:" + str(t))
    total += t
print("avg_acc:" + str(total_acc / iter_time))
print("avg_time:" + str(total / iter_time))