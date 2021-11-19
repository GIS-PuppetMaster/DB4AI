import math

import pyodbc
import torch


class GDBC(object):
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.ans = None

    """
        与数据库的连接，返回ok或任何其它信息。
    """

    def connect(self):
        # 建立连接
        self.conn = pyodbc.connect('DRIVER={GaussMPP};SERVER=127.0.0.1;DATABASE=omm;UID=omm;PWD=Gauss@123')
        self.cursor = self.conn.cursor()

    """
        关闭与数据库的连接。
    """

    def close(self):
        #  关闭连接
        self.cursor.commit()
        self.cursor.close()
        self.conn.close()
        print("Successfully closed!")
        return True

    """
        执行sql。
    """

    def execute(self, sql: str):
        #  执行sql
        try:
            self.cursor.execute(sql)
            self.cursor.commit()
            self.ans = self.cursor.fetchall()
        except :
            self.ans = "no result"
    """
        取数据，返回。
    """

    def fetch(self):
        return self.ans


def str_to_list(data):
    if isinstance(data, str):
        new_string = data.replace("{", "[").replace("}", "]")
        new_data = eval(new_string)
    else:
        new_data = data
    return new_data


if __name__ == '__main__':
    gdbc = GDBC()
    gdbc.connect()
    gdbc.execute("drop table if exists i;")
    gdbc.execute("select data from x")
    a = str_to_list(gdbc.fetch()[0][0])
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
    gdbc.execute("select data from y")
    a = str_to_list(gdbc.fetch()[0][0])
    ly = []
    s = []
    j = 0
    for i in a:
        if j < 30:
            j += 1
            s.append(float(i))
        else:
            j = 1
            lx.append(s)
            s = [float(i)]
    ly.append(s)
    y = torch.tensor(ly,requires_grad=True)
    while True:
        w = torch.ones((30, 1), requires_grad=True)
        print(w)
        a = torch.matmul(x,w)
        hx = 1.0/(1 + torch.exp(-1 * torch.matmul(x,w)))
        loss = 0.0001 * torch.mean(torch.pow(w,2)) - torch.mean(y * torch.log(hx) - (1-y) * torch.log(1-hx))
        loss.backward()
        wgrad = w.grad
        w = w - w.grad * 0.01
    gdbc.close()
