import math
from time import time

import numpy as np
import pyodbc
import torch
execute_time = 0

class GDBC(object):
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.cursor_for_fetch = None
        self.ans = None

    """
        与数据库的连接，返回ok或任何其它信息。
    """

    def connect(self):
        # 建立连接
        self.conn = pyodbc.connect('DRIVER={GaussMPP};SERVER=127.0.0.1;DATABASE=omm;UID=omm;PWD=Gauss@123')
        self.cursor = self.conn.cursor()
        self.cursor_for_fetch = self.conn.cursor()

    """
        关闭与数据库的连接。
    """

    def close(self):
        #  关闭连接
        self.cursor.commit()
        self.cursor.close()
        self.cursor_for_fetch.close()
        self.conn.close()
        print("Successfully closed!")
        return True

    """
        执行sql。
    """

    def execute(self, sql: str, raw_sql=False):
        #  执行sql
        try:
            # self.cursor.commit()
            global execute_time
            if raw_sql or "qp4ai_select" in sql or "qp4ai_if_tensor_exists" in sql:
                t = time()
                self.cursor_for_fetch.execute(sql)
                execute_time += time() - t
                self.ans = self.cursor_for_fetch.fetchall()
            else:
                t = time()
                self.cursor.execute(sql)
                execute_time += time() - t
                if "qp4ai_print_matrix" in sql:
                    self.cursor.commit()
            # if not raw_sql and 'if_tensor_exists' not in sql and 'qp4ai_select' not in sql:
            #     assert self.ans[0][0] == 0
        except:
            raise Exception(f'error sql:{sql}')
            # self.ans = "no result"

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
    i = 0
    gdbc.execute(f"drop table if exists real_multi_class_x")
    gdbc.execute(f"drop table if exists real_multi_class_test_x")
    gdbc.execute(f"drop table if exists real_multi_class_y")
    gdbc.execute(f"drop table if exists real_multi_class_test_y")
    gdbc.execute(f"create table real_multi_class_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float)")
    gdbc.execute(f"create table real_multi_class_test_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float)")
    gdbc.execute(f"create table real_multi_class_y(flag1 int,flag2 int,flag3 int,flag4 int)")
    gdbc.execute(f"create table real_multi_class_test_y(flag1 int,flag2 int,flag3 int,flag4 int)")

    with open('car.txt') as f:
        texts = f.readlines()
        buying = ['vhigh', 'high', 'med', 'low']
        maint = ['vhigh', 'high', 'med', 'low']
        doors = ['2', '3', '4', '5more']
        persons = ['2', '4', 'more']
        lug_boot = ['small', 'med', 'big']
        safety = ['low', 'med', 'high']
        if_ac = ['acc', 'unacc', 'good', 'vgood']
        for i in range(len(texts)):
            print(i)
            text = texts[i].strip().split(',')
            if i % 10 == 0:
                s_x = "real_multi_class_test_x"
                s_y = "real_multi_class_test_y"
            else:
                s_x = "real_multi_class_x"
                s_y = "real_multi_class_y"
            gdbc.execute(f"insert into {s_x} values ({buying.index(text[0]) * 0.25}, {maint.index(text[1]) * 0.25}, {doors.index(text[2]) * 0.25}"
                         f", {persons.index(text[3]) * 0.33}, {lug_boot.index(text[4]) * 0.33}, {safety.index(text[5]) * 0.33})")
            if if_ac.index(text[6]) == 0:
                gdbc.execute(f"insert into {s_y} values (1,0,0,0)")
            if if_ac.index(text[6]) == 1:
                gdbc.execute(f"insert into {s_y} values (0,1,0,0)")
            if if_ac.index(text[6]) == 2:
                gdbc.execute(f"insert into {s_y} values (0,0,1,0)")
            if if_ac.index(text[6]) == 3:
                gdbc.execute(f"insert into {s_y} values (0,0,0,1)")
    # gdbc.execute("drop table if exists real_x")
    # gdbc.execute("drop table if exists real_y")
    # gdbc.execute("drop table if exists real_test_x")
    # gdbc.execute("drop table if exists real_test_y")
    # gdbc.execute("create table real_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float,dim7 float,dim8 float,"
    #              "dim9 float,dim10 float,dim11 float,dim12 float,dim13 float,dim14 float,dim15 float,dim16 float,dim17 float,"
    #              "dim18 float,dim19 float,dim20 float,dim21 float,dim22 float,dim23 float,dim24 float,dim25 float,dim26 float,"
    #              "dim27 float,dim28 float,dim29 float,dim30 float)")
    # gdbc.execute("create table real_test_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float,dim7 float,dim8 float,"
    #              "dim9 float,dim10 float,dim11 float,dim12 float,dim13 float,dim14 float,dim15 float,dim16 float,dim17 float,"
    #              "dim18 float,dim19 float,dim20 float,dim21 float,dim22 float,dim23 float,dim24 float,dim25 float,dim26 float,"
    #              "dim27 float,dim28 float,dim29 float,dim30 float)")
    # gdbc.execute("create table real_y(dim1 float)")
    # gdbc.execute("create table real_test_y(dim1 float)")
    #
    # with open('wdbc.txt','r') as f:
    #     lines = f.readlines()
    #     total = []
    #     for a in lines:
    #         s = []
    #         ta = a.split(',')
    #         for j in range(len(ta)):
    #             if j > 1:
    #                 s.append(float(eval(ta[j])))
    #             else:
    #                 s.append(ta[j])
    #         total.append(s)
    # print(len(total))
    # print(total[0])
    # for i in range(len(total[0])):
    #     if i > 1:
    #         max = 0
    #         min = 99999
    #         for j in range(len(total)):
    #             temp = []
    #             if total[j][i] > max:
    #                 max = total[j][i]
    #             elif total[j][i] < min:
    #                 min = total[j][i]
    #         for j in range(len(total)):
    #             total[j][i] = (total[j][i] - min) / (max - min)
    # print(total[0])
    # for i in range(len(total)):
    #     print(i)
    #     if i < 500:
    #         gdbc.execute(f"insert into real_x values({total[i][2]},{total[i][3]},{total[i][4]},{total[i][5]},{total[i][6]},{total[i][7]},{total[i][8]},{total[i][9]},{total[i][10]},{total[i][11]}"
    #                      f",{total[i][12]},{total[i][13]},{total[i][14]},{total[i][15]},{total[i][16]},{total[i][17]},"
    #                      f"{total[i][18]},{total[i][19]},{total[i][20]},{total[i][21]},{total[i][22]},{total[i][23]},{total[i][24]},{total[i][25]},{total[i][26]},"
    #                      f"{total[i][27]},{total[i][28]},{total[i][29]},{total[i][30]},{total[i][31]})")
    #         if total[i][1] == 'B':
    #             gdbc.execute(f"insert into real_y values(1.0)")
    #         elif total[i][1] == 'M':
    #             gdbc.execute(f"insert into real_y values(0.0)")
    #     else:
    #         gdbc.execute(f"insert into real_test_x values({total[i][2]},{total[i][3]},{total[i][4]},{total[i][5]},{total[i][6]},{total[i][7]},{total[i][8]},{total[i][9]},{total[i][10]},{total[i][11]}"
    #                      f",{total[i][12]},{total[i][13]},{total[i][14]},{total[i][15]},{total[i][16]},{total[i][17]},"
    #                      f"{total[i][18]},{total[i][19]},{total[i][20]},{total[i][21]},{total[i][22]},{total[i][23]},{total[i][24]},{total[i][25]},{total[i][26]},"
    #                      f"{total[i][27]},{total[i][28]},{total[i][29]},{total[i][30]},{total[i][31]})")
    #         if total[i][1] == 'B':
    #             gdbc.execute(f"insert into real_test_y values(1.0)")
    #         elif total[i][1] == 'M':
    #             gdbc.execute(f"insert into real_test_y values(0.0)")
    # gdbc.close()
    g = np.array([[1., 2, 2], [3, 4, 3]])
    print(g)
    a = torch.ones([2, 3], requires_grad=True)
    a = torch.tensor(g, requires_grad=True)
    print(a)
    b = torch.ones([3, 4], requires_grad=True)
    print(b)
    c = torch.ones([1, 7], requires_grad=True)
    print(c)
    d = torch.mean(torch.exp(1 - torch.pow(torch.matmul(a, b) + c, 2)))
    d.backward()
    print(c.grad)
    # e = torch.ones([2,4],requires_grad=True)
    # f = torch.mean(torch.exp(1-torch.pow(torch.matmul(a,b) + e,2)))
    # f.backward()
    # print(e.grad)
