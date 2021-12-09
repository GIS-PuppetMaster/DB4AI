import math
import re
from array import array

import numpy
import numpy as np
import psycopg2
import torch
from functools import wraps
from copy import copy, deepcopy
import sklearn
import unittest
from sklearn import metrics as sk_metrics
import pickle as pk
from gdbc import GDBC

operators = {'Add', 'Sub', 'Mul', 'Div', 'LOG', 'POW', 'SQRT', 'CHOLESKY', 'QR', 'SVD', 'NORM', 'COND', 'DET',
                'RANK', 'TRACE', 'RESHAPE', 'TRANSPOSE', 'SHAPE', 'EXP', 'MATMUL', 'DOT', 'INNER', 'OUTER', 'SUM',
                'TENSORDOT', 'KRON', 'STACK', 'GRADIENT', 'Deepcopy', 'Shallowcopy', 'Argmax', 'Argmin', 'Sign',
                'Slice', 'Relu', 'Tanh', 'Softmax', 'Sigmod', 'Elu', 'Adam', 'MEAN', 'MAX', 'MIN', 'Abs', 'ARGSORT',
                'SORT', 'REVERSE', 'AUC', 'MSE', 'F1', 'Backward', 'ACC', 'RECALL', 'PRECISION', 'WLS', 'REPEAT',
                'UNSQUEEZE', 'CleanGrad', 'Negative', 'TensorFromSql'}


class Tensor:
    def __init__(self, name, cursor, *args, **kwargs):
        self.table = Table(name, cursor)
        self.next = None
        self.grad_fn = None
        self.cursor = cursor

    def __del__(self):
        del self.table
    
    def set_next(self, l: list):
        self.next = l
    
    def get_next(self):
        return self.next

    def clear_next(self):
        self.next = None
    
    def set_grad_fn(self, n):
        self.grad_fn = n

    def get_grad_fn(self):
        return self.grad_fn


class Table:
    def __init__(self, name, cursor, *args, **kwargs):
        self.name = name
        self.cursor = cursor

    def __del__(self):
        pass
        # self.cursor.execute(f"drop table if exists {self.name}")


def preprocessing(fun):
    @wraps(fun)
    def decorated(node, **kwargs):
        # todo 自动类型转换
        for i in range(len(node.vars)):
            if re.fullmatch(re.compile(r'[A-Z]+.*', re.S), node.vars[i]):
                node.vars[i] = "\"" + node.vars[i] + "\""
        # if not node.with_grad and not isinstance(node, GRADIENT):
        #     with torch.no_grad():
        #         return fun(node, **kwargs)
        # else:
        if isinstance(node, Backward):
            node.executor.backward_end = node.id
        # 避免加入backward后变量
        if node.executor.backward_end != 0 and node.id > node.executor.backward_end:
            flag = 0
        else:
            flag = 1
        if node.__class__.__name__ in operators and node.__class__.__name__ not in ['Backward', 'CleanGrad','TensorFromSql'] and flag == 1:
            if node.vars[0] in node.executor.tensor_dict:
                del node.executor.tensor_dict[node.vars[0]]
            node.executor.tensor_dict[node.vars[0]] = Tensor(node.vars[0], node.cursor)
            node.executor.tensor_dict[node.vars[0]].set_next(list(filter(None,list(map(lambda x: node.executor.tensor_dict[x.vars[0]] if x.vars[0] in node.executor.tensor_dict else None, node.pre_nodes())))))
            node.executor.tensor_dict[node.vars[0]].set_grad_fn(node)
        elif node.__class__.__name__ is 'Assignment' and flag == 1:
            if node.vars[0] in node.executor.tensor_dict:
                del node.executor.tensor_dict[node.vars[0]]
            if node.vars[1] in node.executor.tensor_dict:
                node.executor.tensor_dict[node.vars[0]] = Tensor(node.vars[0], node.cursor)
                node.executor.tensor_dict[node.vars[0]].set_next(node.executor.tensor_dict[node.vars[1]].get_next())
                node.executor.tensor_dict[node.vars[0]].set_grad_fn(node.executor.tensor_dict[node.vars[1]].get_grad_fn())
        if len(node.vars) != 0 and flag == 1:
            node.executor.var_dict[node.vars[0]] = node

        # 清空记录梯度的表：
        # 表'grad_' + str(node.id)为该节点最终梯度(算子本身梯度乘以上游梯度)的单个返回值(仅有一个返回值时）
        # 表'grad_' + str(node.id) + '_1/2'为该节点最终梯度(算子本身梯度乘以上游梯度)的多个返回值(返回值大于2，如mul、matmul)
        # 表'grad_' + str(node.id) + '_temp1/2'为计算最终梯度的中间结果(通常是算子本身梯度，需乘以上游梯度)
        # 表'grad_' + str(node.id)为算子输入的上游梯度

        node.cursor.execute(f"drop table if exists {'grad_' + str(node.id)};"
                            f"drop table if exists {'grad_input_' + str(node.id)};"
                            f"drop table if exists {'grad_' + str(node.id) + '_1'};"
                            f"drop table if exists {'grad_' + str(node.id) + '_2'};"
                            f"drop table if exists {'grad_' + str(node.id) + '_temp1'};"
                            f"drop table if exists {'grad_' + str(node.id) + '_temp2'};")
        return fun(node, **kwargs)

    return decorated


def dump_tensor(tensors: dict, path: str):
    with open(path, 'wb') as f:
        pk.dump(tensors, f)


def fill_slice_var(slice_index, cursor):
    s = copy(slice_index)
    for idx in range(len(s)):
        if isinstance(s[idx], str):
            cursor.execute(f"select data from {s[idx]};")
            s[idx] = str_to_list(cursor.fetch()[0][0])[0]
        elif isinstance(s[idx], slice):
            start = s[idx].start
            step = s[idx].step
            stop = s[idx].stop
            if isinstance(s[idx].start, str):
                cursor.execute(f"select data from {s[idx].start};")
                start = str_to_list(cursor.fetch()[0][0])[0]
            if isinstance(s[idx].step, str):
                cursor.execute(f"select data from {s[idx].step};")
                step = str_to_list(cursor.fetch()[0][0])[0]
            if isinstance(s[idx].stop, str):
                cursor.execute(f"select data from {s[idx].stop};")
                stop = str_to_list(cursor.fetch()[0][0])[0]
            s[idx] = slice(start, stop, step)
    return tuple(s)


def load_tensor(path):
    with open(path, 'rb') as f:
        return pk.load(f)


def parse_slice(slice_info):
    total_slice = []
    for idx in slice_info:
        idx = idx.strip()
        if idx == ':':
            total_slice.append(slice(None, None, None))
        elif idx == '...':
            total_slice.append(...)
        elif ':' in idx:
            total_slice.append(slice(
                *list(map(lambda x: None if x == '' else (str(x) if re.fullmatch(re.compile(r'[a-zA-Z]+.*', re.S), x)
                                                          else int(x)), idx.split(':')))))
        else:
            if re.fullmatch(re.compile(r'([a-zA-Z_]+[a-zA-Z0-9_]*)', re.S), idx):
                total_slice.append(idx)
            else:
                total_slice.append(int(idx))
    return total_slice


def str_to_list(data):
    if isinstance(data, str):
        new_string = data.replace("{", "[").replace("}", "]")
        new_data = eval(new_string)
    else:
        new_data = data
    return new_data






class DimensionError(Exception):
    pass


class DivZeroError(Exception):
    pass


class Node:
    # 计算图中节点类的父类
    def __init__(self, type_id, with_grad=False, physic_algorithm='tensor', **kwargs):
        self.physic_algorithm = physic_algorithm
        self.id = kwargs['id']
        self.type_id = type_id
        self.with_grad = with_grad
        self.out_edges = []
        self.in_edges = []
        self.input_data_edges = []
        self.branches = kwargs['branches']
        self.branches_set = None
        self.vars = []
        self.executor = None
        self.batch_counter = 0
        self._default_batch_size = 0
        self.batch_size = 0
        self.use_batch = True
        self.fathers = list(set([edge.start for edge in self.in_edges]))
        self.sons = list(set([edge.end for edge in self.out_edges]))
        self.release_list = []
        self.in_loop = -1
        self.finished = False
        self.visited_sequence = []
        self.grad = None
        '''# self.conn = psycopg2.connect(database="postgres", user="postgres", host="114.115.156.203", port="2333")
        self.cursor = # self.conn.cursor()'''
        # self.conn = None
        self.cursor = None
        # if 'cursor' in kwargs.keys():
        #     self.cursor = kwargs['cursor']
        # else:
        #     self.set_conn()

    @property
    def default_batch_size(self):
        return self._default_batch_size

    @default_batch_size.setter
    def default_batch_size(self, value):
        if self.use_batch:
            if value < 1:
                # 输入的是按比例划分batch_size
                self.batch_size = int(self.out_edges[0].data_shape[0] * value)
            else:
                assert isinstance(value, int)
                self.batch_size = value
        else:
            self.batch_size = self.out_edges[0].data_shape[0]

    def GetId(self):
        return self.id

    def generate_data_edges(self):
        for in_edge in self.in_edges:
            if isinstance(in_edge.start, Loop) or isinstance(in_edge.start, If):
                for data_edge in in_edge.start.in_edges:
                    if in_edge.var == data_edge.var:
                        self.input_data_edges.append(data_edge)
            else:
                self.input_data_edges.append(in_edge)

    @preprocessing
    def run(self, **kwargs):
        pass

    def next_nodes(self):
        self.sons = list(set([edge.end for edge in self.out_edges]))
        return self.sons

    def pre_nodes(self):
        self.fathers = list(set([edge.start for edge in self.in_edges]))
        return self.fathers

    def infer_data(self):
        pass

    def GetType(self):
        return self.type_id

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.id == other.id) and (self.type_id == other.type_id)
        else:
            return False

    def __hash__(self):
        return hash(self.id + self.type_id)

    def __call__(self, executor):
        pass

    def set_vars(self, input):
        if isinstance(input, list):
            self.vars = input
        else:
            self.vars.append(input)

    def get_vars(self):
        return self.vars

    # 用于run方法的域设置
    # def set_conn(self):
    #     # self.cursor = # self.conn.cursor()
    #     self.cursor = GDBC()
    #     self.cursor.connect()

    def __repr__(self):
        return f'id:{self.id}, branches:{self.branches}, vars:{self.vars}'

    def op_broadcast(self, op, input_table_1, input_table_2, output_table):
        self.cursor.execute(f"drop table if exists {output_table}")
        self.cursor.execute(f"select rows,cols from {input_table_1}")
        result1 = self.cursor.fetch()
        self.cursor.execute(f"select rows,cols from {input_table_2}")
        result2 = self.cursor.fetch()
        try:
            if result1[0][0] == result2[0][0] and result1[0][1] == result2[0][1]:
                # 同维度矩阵特殊情况，无需广播,对应元素值依次操作
                self.fun_opt(op, input_table_1, input_table_2, output_table)
            # 行相等，列不相等，则有一组矩阵列数为1，否则不满足广播条件
            elif result1[0][0] == result2[0][0]:
                self.cursor.execute(f"select data from {input_table_1}")
                data1 = str_to_list(self.cursor.fetch()[0][0])
                self.cursor.execute(f"select data from {input_table_2}")
                data2 = str_to_list(self.cursor.fetch()[0][0])
                new_data = []
                if result1[0][1] == 1:
                    for i in data2:
                        new_data.append(self.val_opt(op, data1[0], i))
                    self.cursor.execute(f"create table {output_table}(rows int, cols int,trans int,data double "
                                        f"precision[] )")
                    self.cursor.execute(
                        f"insert into {output_table} values({result2[0][0]}, {result2[0][1]}, 0, array{new_data})")
                else:
                    for i in data1:
                        new_data.append(self.val_opt(op, i, data2[0]))
                    self.cursor.execute(f"create table {output_table}(rows int, cols int,trans int,data double "
                                        f"precision[] )")
                    self.cursor.execute(
                        f"insert into {output_table} values({result1[0][0]}, {result1[0][1]}, 0, array{new_data})")
            # 行不相等，列相等，则有一组矩阵行数为1，否则不满足广播条件
            elif result1[0][0] != result2[0][0] and result1[0][1] == result2[0][1]:
                self.cursor.execute(f"select data from {input_table_1}")
                data1 = str_to_list(self.cursor.fetch()[0][0])
                self.cursor.execute(f"select data from {input_table_2}")
                data2 = str_to_list(self.cursor.fetch()[0][0])
                new_data = []
                if result1[0][0] == 1:
                    for i in range(len(data1)):
                        for j in range(len(data2)):
                            if i == j % len(data1):
                                new_data.append(self.val_opt(op, data1[i], data2[j]))
                    self.cursor.execute(f"create table {output_table}(rows int, cols int,trans int,data double "
                                        f"precision[] )")
                    self.cursor.execute(
                        f"insert into {output_table} values({result2[0][0]}, {result2[0][1]}, 0, array{new_data})")
                elif result2[0][0] == 1:
                    for i in range(len(data1)):
                        for j in range(len(data2)):
                            if j == i % len(data2):
                                new_data.append(self.val_opt(op, data1[i], data2[j]))
                    self.cursor.execute(f"create table {output_table}(rows int, cols int,trans int,data double "
                                        f"precision[] )")
                    self.cursor.execute(
                        f"insert into {output_table} values({result1[0][0]}, {result1[0][1]}, 0, array{new_data})")
                else:
                    raise DimensionError()
            # 矩阵形状的各个维度不一,则各个维度上必有一值为1
            elif result1[0][0] != result2[0][0] and result1[0][1] != result2[0][1]:
                self.cursor.execute(f"select data from {input_table_1}")
                data1 = str_to_list(self.cursor.fetch()[0][0])
                self.cursor.execute(f"select data from {input_table_2}")
                data2 = str_to_list(self.cursor.fetch()[0][0])
                new_data = []
                if result1[0][0] == 1 and result1[0][1] == 1:
                    for i in range(len(data2)):
                        new_data.append(self.val_opt(op, data1[0], data2[i]))
                    self.cursor.execute(f"create table {output_table}(rows int, cols int,trans int,data double "
                                        f"precision[] )")
                    self.cursor.execute(
                        f"insert into {output_table} values({result2[0][0]}, {result2[0][1]}, 0, array{new_data})")
                elif result2[0][0] == 1 and result2[0][1] == 1:
                    for i in range(len(data1)):
                        new_data.append(self.val_opt(op, data1[i], data2[0]))
                    self.cursor.execute(f"create table {output_table}(rows int, cols int,trans int,data double "
                                        f"precision[] )")
                    self.cursor.execute(
                        f"insert into {output_table} values({result1[0][0]}, {result1[0][1]}, 0, array{new_data})")
                # 1行n列与m行1列做运算
                elif result1[0][0] == 1 and result2[0][1] == 1:
                    for i in range(len(data2)):
                        for j in range(len(data1)):
                            new_data.append(self.val_opt(op, data1[j], data2[i]))
                    self.cursor.execute(f"create table {output_table}(rows int, cols int,trans int,data double "
                                        f"precision[] )")
                    self.cursor.execute(
                        f"insert into {output_table} values({result2[0][0]}, {result1[0][1]}, 0, array{new_data})")
                elif result1[0][1] == 1 and result2[0][0] == 1:
                    for i in range(len(data1)):
                        for j in range(len(data2)):
                            new_data.append(self.val_opt(op, data1[i], data2[j]))
                    self.cursor.execute(f"create table {output_table}(rows int, cols int,trans int,data double "
                                        f"precision[] )")
                    self.cursor.execute(
                        f"insert into {output_table} values({result1[0][0]}, {result2[0][1]}, 0, array{new_data})")
                else:
                    raise DimensionError()
            else:
                pass
        except DimensionError:
            print("矩阵不匹配，无法进行广播操作")

    def val_opt(self, op, i, j):
        if op == 'add':
            return i + j
        elif op == 'sub':
            return i - j
        elif op == 'mul':
            return i * j
        elif op == 'div':
            return i / j

    def fun_opt(self, op, input_table_1, input_table_2, output_table):
        if op == 'add':
            self.cursor.execute(f"select db4ai_add('{input_table_1}', '{input_table_2}', '{output_table}');")
        elif op == 'sub':
            self.cursor.execute(f"select db4ai_sub('{input_table_1}', '{input_table_2}', '{output_table}');")
        elif op == 'mul':
            self.cursor.execute(f"select db4ai_mul('{input_table_1}', '{input_table_2}', '{output_table}');")
        elif op == 'matmul':
            self.cursor.execute(f"select db4ai_matmul('{input_table_1}', '{input_table_2}', '{output_table}');")

    def is_equal_shape(self, input_table_1, input_table_2):
        self.cursor.execute(f"select rows,cols from {input_table_1}")
        shape1 = self.cursor.fetch()
        self.cursor.execute(f"select rows,cols from {input_table_2}")
        shape2 = self.cursor.fetch()
        if shape1[0][0] == shape2[0][0] and shape1[0][1] == shape2[0][1]:
            return True
        else:
            return False

    def groupby(self, table_name_1,table_name_2,rows,cols):
        self.cursor.execute(f"select rows,cols from {table_name_1}")
        shape = self.cursor.fetch()[0]
        self.cursor.execute(f"select data from {table_name_1}")
        data = str_to_list(self.cursor.fetch()[0][0])
        if shape[0] != rows or shape[1] != cols:
            if rows == 1 and cols != 1:
                self.cursor.execute(f"select db4ai_sum('{table_name_1}', 0, '{table_name_2}');")
            elif cols == 1 and rows != 1:
                self.cursor.execute(f"select db4ai_sum('{table_name_1}', 1, '{table_name_2}');")
            else:
                # TODO:sum算子对所有元素求和
                sum = 0
                for i in range(len(data)):
                    sum += data[i]
                self.cursor.execute(f"drop table if exists {table_name_2}")
                self.cursor.execute(f"create table {table_name_2}(rows int, cols int,trans int,data double "
                                    f"precision[] )")
                self.cursor.execute(
                    f"insert into {table_name_2} values({rows}, {cols}, 0, array{[sum]})")
        else:
            self.cursor.execute(f"select * into {table_name_2} from {table_name_1}")

# 通过继承实现的其它节点类


class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)


# 创建张量所用节点
class CreateTensor(Node):
    def __init__(self, data_shape, var, **kwargs):
        super().__init__(1, **kwargs)
        self.set_vars(var)
        self.grad = None

    @preprocessing
    def run(self, **kwargs):
        pass


# 该类用来存储常量，常见如constant.PI、constant.E
class Val(Node):
    def __init__(self, var, val, **kwargs):
        super().__init__(2, **kwargs)
        self.value = val
        if isinstance(var, list):
            self.vars = var
        elif var is None:
            self.vars = []
        else:
            self.vars = [var]

    def set_val(self, value):
        self.value = value

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"DROP TABLE IF EXISTS {self.vars[0]}")
        self.cursor.execute(f"create table {self.vars[0]}(rows int,cols int,trans int,data double precision[])")
        self.cursor.execute(f"insert into {self.vars[0]} values (1,1,0, array{[self.value]})")
        # self.conn.commit()

    def get_val(self):
        return self.value

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = (1,)


class TensorFromSql(Node):
    # TODO: 注册算子@樊宣伯
    def __init__(self, **kwargs):
        super().__init__(3, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # TODO：代码实现@路亚彬, @樊宣伯
        self.cursor.execute(f"select db4ai_load('{self.vars[1]}','{self.vars[0]}')")


class Sql(Node):
    def __init__(self, t_info, var, **kwargs):
        super().__init__(3, **kwargs)
        self.t_search_sentences = t_info
        self.vars = var
        self.shape = None

    @preprocessing
    def run(self, **kwargs):
        # TODO：执行SQL@路亚彬
        pass


class Random(Node):
    def __init__(self, boundary, data_shape, distribution, var, **kwargs):
        super().__init__(4, **kwargs)
        self.boundary = boundary
        self.vars = var
        self.data_shape = data_shape
        # 记录data shape和boundary中可能出现的变量名
        self.data_shape_var = {}
        self.boundary_var = {}
        self.dis_args = [0, 1]
        distribution_list = ['normal', 'Uniform', 'Bernoulli']

        if self.boundary != '':
            self.distribution = 1
            self.dis_args = [eval(self.boundary)[0], eval(self.boundary)[1]]
        elif distribution == '' or distribution is None or (isinstance(distribution, list) and len(distribution) == 0):
            self.distribution = 0
        else:
            self.distribution = distribution_list.index(distribution)

    @preprocessing
    def run(self, **kwargs):
        #     # TODO:
        # 如果data shape中包含var

        self.cursor.execute(f"drop table if exists {self.vars[0]}")
        if isinstance(self.data_shape, str):
            # 运行时使用变量的值填充变量名
            for name in self.data_shape_var.keys():
                self.cursor.execute(f"select data from {name};")
                self.data_shape_var[name] = str_to_list(self.cursor.fetch()[0][0])[0]
                # 转换
            self.data_shape = eval(self.data_shape, self.data_shape_var)
        if self.distribution not in [0, 1, 2]:
            distribution_list = ['normal', 'Uniform', 'Bernoulli']
            self.distribution = distribution_list.index(self.distribution)
        self.cursor.execute(
            f"select db4ai_random({self.data_shape[0]}, {self.data_shape[1]}, {self.distribution}, {self.dis_args[0]}, {self.dis_args[1]}, 0, '{self.vars[0]}')")
        # self.conn.commit()

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_type = 'ndarray'
            edge.data_shape = self.data_shape

    def handle_include_var(self, **change_info):
        if change_info['d_has_var'] and change_info['b_has_var']:
            self.data_shape_var = change_info['d_var']
            self.boundary_var = change_info['b_var']
        elif change_info['d_has_var']:
            self.data_shape_var = change_info['d_var']
            self.boundary = eval(self.boundary)
        elif change_info['b_has_var']:
            self.boundary_var = change_info['b_var']
            self.data_shape = eval(self.data_shape)
        else:
            self.data_shape = eval(self.data_shape)
            self.boundary = eval(self.boundary)

    def set_dis_args(self, args):
        self.dis_args = args


# 逻辑控制所用节点
class Loop(Node):
    def __init__(self, condition, loop_id, **kwargs):
        super().__init__(5, **kwargs)
        if condition or isinstance(condition, str):
            self.dead_cycle = condition
            self.times = -1
        else:
            self.dead_cycle = False
            self.times = condition
        self.loop_id = loop_id
        assert self.loop_id == self.id
        self.loop_pair = None

    @preprocessing
    def run(self, **kwargs):
        visited = kwargs['visited']
        executor = kwargs['executor']
        if self.loop_pair in visited:
            visited.remove(self.loop_pair)
        self.loop_pair.finished = False
        self.times += 1

    def next_nodes(self):
        assert self.loop_pair is not None
        end_nodes = [edge.end for edge in self.out_edges]
        if self.loop_pair in end_nodes:
            end_nodes.remove(self.loop_pair)
        if isinstance(self.dead_cycle, str):
            self.cursor.execute(f"select data from {self.dead_cycle};")
            self.dead_cycle = str_to_list(self.cursor.fetch()[0][0])[0]
        # 循环结束
        if self.dead_cycle <= self.times:
            # 找到对应的Loop_End
            self.executor.finished_loop_id.add(self.loop_id)
            self.loop_pair.return_next = True
            return [self.loop_pair]
        else:
            return end_nodes


class LoopEnd(Node):
    def __init__(self, loop_id, **kwargs):
        super().__init__(6, **kwargs)
        self.loop_id = loop_id
        self.loop_pair = None
        self.return_next = False

    @preprocessing
    def run(self, **kwargs):
        visited = kwargs['visited']
        executor = kwargs['executor']
        # 从visited中删除对应的LoopEnd
        visited.remove(self.loop_pair)
        self.loop_pair.finished = False
        # 移除loop内的节点
        nodes_in_loop = []
        for node in visited:
            if self.loop_pair.id in node.branches_set:
                nodes_in_loop.append(node)
        for node in nodes_in_loop:
            if node in visited:
                visited.remove(node)
            node.finished = False
            if isinstance(node, LoopEnd):
                node.return_next = False

    def next_nodes(self):
        assert self.loop_pair is not None
        end_nodes = [edge.end for edge in self.out_edges]
        end_nodes.remove(self.loop_pair)

        # 退出循环
        if self.loop_id in self.executor.finished_loop_id:
            # self.executor.finished_loop_id.remove(self.loop_id)
            return end_nodes
        # 继续下一次循环
        else:
            return [self.loop_pair]


class Break(Node):
    def __init__(self, loop_id, **kwargs):
        super().__init__(7, **kwargs)
        self.loop_id = loop_id
        self.loop_pair = None

    def next_nodes(self):
        assert self.loop_pair is not None
        self.executor.finished_loop_id.add(self.loop_id)
        return [self.loop_pair]


class If(Node):
    def __init__(self, **kwargs):
        super().__init__(8, **kwargs)

    def next_nodes(self):
        for edge in self.out_edges:
            para = {}
            for var_name, var_node in edge.need_var:
                self.cursor.execute(f"select data from {var_node.vars[0]}")
                result = str_to_list(self.cursor.fetch()[0][0])
                if len(result) == 1:
                    para[var_name] = result[0]
                else:
                    if not isinstance(result, list):
                        para[var_name] = list(result)
                    else:
                        para[var_name] = result
            res = eval(edge.condition, para)
            if edge.reverse:
                res = not res
            if res:
                return [edge.end]


class IfBranch(Node):
    def __init__(self, **kwargs):
        super().__init__(9, **kwargs)
        self.end_if_pair = None

    @preprocessing
    def run(self, **kwargs):
        assert self.end_if_pair is not None
        self.end_if_pair.selected_branch = self.id

    def next_nodes(self):
        if self.out_edges[0].condition is None:
            return self.sons
        else:
            for edge in self.out_edges:
                para = {}
                for var_name, var_node in edge.need_var:
                    self.cursor.execute(f"select data from {var_node.vars[0]}")
                    para[var_name] = str_to_list(self.cursor.fetch()[0][0])[0]
                res = eval(edge.condition, para)
                if edge.reverse:
                    res = not res
                if res:
                    return [edge.end]


class IfEnd(Node):
    def __init__(self, **kwargs):
        super().__init__(10, **kwargs)
        self.selected_branch = None


class Assignment(Node):
    def __init__(self, var_li, **kwargs):
        super().__init__(11, **kwargs)
        self.vars = var_li
        self.update = False
        self._slice = None
        if 'slice' in kwargs.keys():
            self.slice = kwargs['slice']
        if 'update' in kwargs.keys():
            self.update = True

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, slice_info):
        total_slice = parse_slice(slice_info)
        if len(total_slice) > 0:
            self._slice = total_slice

    @preprocessing
    def run(self, **kwargs):
        if self.vars[0] == '__0loss':
            self.cursor.execute(f"select count(*) from pg_class where relname = '{self.vars[0]}';")
            rows = self.cursor.fetch()
            if rows[0][0] == 1:
                self.cursor.execute(f"select data from {self.vars[0]};")
                data = str_to_list(self.cursor.fetch()[0][0])[0]
                self.cursor.execute(f"select data from {self.vars[1]}")
                new_data = str_to_list(self.cursor.fetch()[0][0])[0]
                print("delta: " + str(new_data-data))

        self.cursor.execute(f"select count(*) from pg_class where relname = '{self.vars[1]}';")
        rows = self.cursor.fetch()
        flag_right = rows[0][0] == 1
        if self.vars[0] in ['auc','acc','recall', 'prec', 'mse','f1'] and flag_right is False:
            print(self.vars[0])
            pass
        else:
            assert flag_right is True
            if self.slice is None:
                self.cursor.execute(f"DROP TABLE IF EXISTS {self.vars[0]}")
                self.cursor.execute(f"select * into {self.vars[0]} from {self.vars[1]}")
            else:
                self.cursor.execute(f"select data from {self.vars[1]}")
                new_data = str_to_list(self.cursor.fetch()[0][0])[0]
                self.cursor.execute(f"select rows,cols from {self.vars[0]}")
                shape = self.cursor.fetch()
                self.cursor.execute(f"select data from {self.vars[0]}")
                old_data = str_to_list(self.cursor.fetch()[0][0])
                s = fill_slice_var(self.slice, self.cursor)
                try:
                    if len(s) == 2:
                        if s[0] is ... and s[1] is ...:
                            for i in range(len(old_data)):
                                old_data[i] = new_data
                        elif s[0] is ...:
                            for i in range(shape[0][0]):
                                old_data[i * shape[0][1] + s[1]] = new_data
                        elif s[1] is ...:
                            for i in range(shape[0][1]):
                                old_data[s[0] * shape[0][1] + i] = new_data
                        else:
                            old_data[s[0] * shape[0][1] + s[1]] = new_data
                    elif len(s) == 1:
                        old_data[s[0]] = new_data
                except IndexError:
                    print(old_data)
                    print(shape)
                self.cursor.execute(f"update {self.vars[0]} set data = array{old_data};")


class Add(Node):
    def __init__(self, **kwargs):
        super().__init__(12, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.op_broadcast("add", self.vars[1], self.vars[2], self.vars[0])

    # 无需重构
    def backward(self, grad_output=1):
        return grad_output, grad_output


class Sub(Node):
    def __init__(self, **kwargs):
        super().__init__(13, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.op_broadcast("sub", self.vars[1], self.vars[2], self.vars[0])

    def backward(self, grad_output=1):
        table_name = 'grad_' + str(self.id)
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        if grad_output == 1:
            s = table_name
        if grad_output != 1:
            s = table_name_temp1
        self.cursor.execute(f"drop table if exists {s}")
        self.cursor.execute(f"create table {s}(rows int, cols int,trans int,data double precision[])")
        self.cursor.execute(f"insert into {s} values (1,1,0,array{[-1]})")
        if grad_output != 1:
            self.op_broadcast("mul", s, grad_output, table_name)
        return grad_output, table_name


class Mul(Node):
    def __init__(self, **kwargs):
        super().__init__(14, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.op_broadcast("mul", self.vars[1], self.vars[2], self.vars[0])
        # self.conn.commit()

    def backward(self, grad_output=1):
        table_name_1 = 'grad_' + str(self.id) + '_1'
        table_name_2 = 'grad_' + str(self.id) + '_2'
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        table_name_temp2 = 'grad_' + str(self.id) + '_temp2'
        self.cursor.execute(f"select rows,cols from {self.vars[1]}")
        shape_1 = self.cursor.fetch()
        self.cursor.execute(f"select rows,cols from {self.vars[2]}")
        shape_2 = self.cursor.fetch()

        # 分类讨论
        if shape_1[0][0] == 1 and shape_1[0][1] == 1:
            flag = 1
        elif shape_2[0][0] == 1 and shape_2[0][1] == 1:
            flag = 2
        elif shape_1 == shape_2:
            flag = 3
        if grad_output == 1:
            s_1 = table_name_1
            s_2 = table_name_2
        else:
            s_1 = table_name_temp1
            s_2 = table_name_temp2
        self.cursor.execute(f"drop table if exists {s_1}")
        self.cursor.execute(f"drop table if exists {s_2}")
        self.cursor.execute(f"select data from {self.vars[1]}")
        data_1 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select data from {self.vars[2]}")
        data_2 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"create table {s_1}(rows int,cols int,trans int,data double precision[])")
        self.cursor.execute(f"create table {s_2}(rows int,cols int,trans int,data double precision[])")

        # 乘以单个常数值
        if flag == 1:
            sum = 0
            for i in data_2:
                sum += i
            self.cursor.execute(f"insert into {s_1} values({shape_1[0][0]},{shape_1[0][1]},0,array{[sum]})")
            self.cursor.execute(f"insert into {s_2} values(1,1,0,array{data_1})")
        elif flag == 2:
            self.cursor.execute(f"insert into {s_1} values(1,1,0,array{[data_2[0]]})")
            sum = 0
            for i in data_1:
                sum += i
            self.cursor.execute(f"insert into {s_2} values({shape_2[0][0]},{shape_2[0][1]},0,array{[sum]})")

        # 矩阵形状相同
        elif flag == 3:
            self.cursor.execute(f"insert into {s_1} values({shape_1[0][0]},{shape_1[0][1]},0,array{data_2})")
            self.cursor.execute(f"insert into {s_2} values({shape_2[0][0]},{shape_2[0][1]},0,array{data_1})")

        if grad_output != 1:
            self.op_broadcast("mul", s_1, grad_output, table_name_1)
            self.op_broadcast("mul", s_2, grad_output, table_name_2)
        return table_name_1, table_name_2


class Div(Node):
    def __init__(self, **kwargs):
        super().__init__(15, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select data from {self.vars[1]}")
        data_1 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select data from {self.vars[2]}")
        data_2 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select rows,cols from {self.vars[1]}")
        shape_1 = self.cursor.fetch()
        self.cursor.execute(f"select rows,cols from {self.vars[2]}")
        shape_2 = self.cursor.fetch()
        self.cursor.execute(f"drop table if exists {self.vars[0]}")
        if shape_1 == shape_2:
            self.cursor.execute(f"select db4ai_div('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        elif len(data_1) == 1:
            new_data = []
            for i in range(len(data_2)):
                new_data.append(data_1[0] / data_2[i])
            self.cursor.execute(f"create table {self.vars[0]}(rows int,cols int,trans int,data double precision[])")
            self.cursor.execute(f"insert into {self.vars[0]} values ({shape_2[0][0]},{shape_2[0][1]},0,array{new_data})")
        elif len(data_2) == 1:
            new_data = []
            for i in range(len(data_1)):
                new_data.append(data_1[i] / data_2[0])
            self.cursor.execute(f"create table {self.vars[0]}(rows int,cols int,trans int,data double precision[])")
            self.cursor.execute(f"insert into {self.vars[0]} values ({shape_1[0][0]},{shape_1[0][1]},0,array{new_data})")


    def backward(self, grad_output=1):
        table_name_1 = 'grad_' + str(self.id) + '_1'
        table_name_2 = 'grad_' + str(self.id) + '_2'
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        table_name_temp2 = 'grad_' + str(self.id) + '_temp2'
        if grad_output == 1:
            s_1 = table_name_1
            s_2 = table_name_2
        else:
            s_1 = table_name_temp1
            s_2 = table_name_temp2
        self.cursor.execute(f"drop table if exists {s_1}")
        self.cursor.execute(f"drop table if exists {s_2}")
        self.cursor.execute(f"create table {s_1}(rows int,cols int,trans int,data double precision[])")
        self.cursor.execute(f"create table {s_2}(rows int,cols int,trans int,data double precision[])")

        self.cursor.execute(f"select data from {self.vars[1]}")
        data_1 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select data from {self.vars[2]}")
        data_2 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select rows,cols from {self.vars[1]}")
        shape_1 = self.cursor.fetch()
        self.cursor.execute(f"select rows,cols from {self.vars[2]}")
        shape_2 = self.cursor.fetch()
        new_data_1 = []
        new_data_2 = []
        if len(data_1) == 1:
            sum = 0
            for i in range(len(data_2)):
                sum += 1 / data_2[i]
                new_data_2.append(data_1[0] * -1 / pow(data_2[i], 2))
            self.cursor.execute(f"insert into {s_1} values (1,1,0,array{[sum]})")
            self.cursor.execute(f"insert into {s_2} values ({shape_2[0][0]},{shape_2[0][1]},0,array{new_data_2})")
        elif len(data_2) == 1:
            sum = 0
            for i in range(len(data_1)):
                new_data_1.append(1 / data_2[0])
                sum += data_1[i] * -1 / pow(data_2[0], 2)
            self.cursor.execute(f"insert into {s_1} values ({shape_1[0][0]},{shape_1[0][1]},0,array{new_data_1})")
            self.cursor.execute(f"insert into {s_2} values (1,1,0,array{[sum]})")
        else:
            for i in range(len(data_1)):
                new_data_1.append(1 / data_2[i])
                new_data_2.append(data_1[i] * -1 / pow(data_2[i], 2))
            self.cursor.execute(f"insert into {s_1} values ({shape_1[0][0]},{shape_1[0][1]},0,array{new_data_1})")
            self.cursor.execute(f"insert into {s_2} values ({shape_2[0][0]},{shape_2[0][1]},0,array{new_data_2})")
        if grad_output != 1:
            self.op_broadcast("mul", s_1, grad_output, table_name_1)
            self.op_broadcast("mul", s_2, grad_output, table_name_2)
        return table_name_1, table_name_2


class LOG(Node):
    def __init__(self, **kwargs):
        super().__init__(16, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # TODO: log疑似不支持多行多列
        # self.cursor.execute(f"select db4ai_log('{self.vars[1]}', '{self.vars[0]}');")
        self.cursor.execute(f"drop table if exists {self.vars[0]};")
        self.cursor.execute(f"select * into {self.vars[0]} from {self.vars[1]};")
        self.cursor.execute(f"select data from {self.vars[1]};")
        ss = str_to_list(self.cursor.fetch()[0][0])
        for i in range(len(ss)):
            ss[i] = math.log(ss[i], math.e)
        self.cursor.execute(f"update {self.vars[0]} set data = array{ss};")


    def backward(self, grad_output=1):
        table_name = 'grad_' + str(self.id)
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        if grad_output == 1:
            s_1 = table_name
        else:
            s_1 = table_name_temp1
        self.cursor.execute(f"drop table if exists {s_1}")
        self.cursor.execute(f"select rows,cols from {self.vars[1]};")
        shape = self.cursor.fetch()[0]
        self.cursor.execute(f"select data from {self.vars[1]};")
        data = str_to_list(self.cursor.fetch()[0][0])
        new_row = []
        for i in data:
            r = 1.0 / i
            new_row.append(r)
        self.cursor.execute(f"create table {s_1}(rows int,cols int,trans int,data double precision[])")
        self.cursor.execute(f"insert into {s_1} values ({shape[0]},{shape[1]},0,array{new_row})")
        # self.conn.commit()
        if grad_output != 1:
            self.op_broadcast("mul", s_1, grad_output, table_name)
        return table_name


class POW(Node):
    def __init__(self, **kwargs):
        super().__init__(17, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select data from {self.vars[2]};")
        pow_exp = str_to_list(self.cursor.fetch()[0][0])
        if len(pow_exp) > 1:
            self.cursor.execute(f"select data from {self.vars[1]};")
            base = str_to_list(self.cursor.fetch()[0][0])[0]
            self.cursor.execute(f"select db4ai_pow_table('{self.vars[2]}', {base}, '{self.vars[0]}');")
        else:
            self.cursor.execute(f"select db4ai_pow('{self.vars[1]}', {pow_exp[0]}, '{self.vars[0]}');")

    def backward(self, grad_output=1):
        table_name = 'grad_' + str(self.id)
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        if grad_output == 1:
            s_1 = table_name
        else:
            s_1 = table_name_temp1
        self.cursor.execute(f"select data from {self.vars[1]};")
        data_1 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select data from {self.vars[2]};")
        data_2 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select data from {self.vars[0]};")
        data_0 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select rows,cols from {self.vars[0]};")
        shape = self.cursor.fetch()[0]
        new_data = []

        if len(data_2) > 1:
            for i in range(len(data_0)):
                new_data.append(data_0[i] * math.log(data_1[0]))
        else:
            for i in range(len(data_0)):
                new_data.append(data_0[i] * data_2[0] / data_1[i])
        self.cursor.execute(f"drop table if exists {s_1}")
        self.cursor.execute(f"create table {s_1}(rows int,cols int,trans int,data double precision[])")
        self.cursor.execute(f"insert into {s_1} values({shape[0]},{shape[1]},0,array{new_data});")
        if grad_output != 1:
            self.op_broadcast("mul", s_1, grad_output, table_name)
        return table_name


class SQRT(Node):
    def __init__(self, **kwargs):
        super().__init__(18, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_sqrt('{self.vars[1]}', '{self.vars[0]}');")
        # self.conn.commit()

    def backward(self, grad_output=1):
        table_name_1 = 'grad_' + str(self.id) + '_1'
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        if grad_output == 1:
            s_1 = table_name_1
        else:
            s_1 = table_name_temp1
        self.cursor.execute(f"drop table if exists {s_1}")
        self.cursor.execute(f"select data from {self.vars[0]};")
        data_2 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select rows,cols from {self.vars[0]};")
        shape = self.cursor.fetch()
        new_data = []
        try:
            for i in range(shape[0][0] * shape[0][1]):
                if data_2[i] != 0:
                    new_data.append(1 / (2 * data_2[i]))
                else:
                    raise DivZeroError()
        except DivZeroError:
            print("除数不可为0")
        self.cursor.execute(f"create table {s_1}(rows int,cols int,trans int,data double precision[])")
        self.cursor.execute(f"insert into {s_1} values ({shape[0][0]},{shape[0][1]},0,array{new_data});")
        # self.conn.commit()
        if grad_output != 1:
            self.op_broadcast("mul", s_1, grad_output, table_name_1)
        return table_name_1


class MATMUL(Node):
    def __init__(self, **kwargs):
        super().__init__(19, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_matmul('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        # self.conn.commit()


    def backward(self, grad_output=1):
        table_name_1 = 'grad_' + str(self.id) + '_1'
        table_name_2 = 'grad_' + str(self.id) + '_2'
        self.cursor.execute(f"drop table if exists {table_name_1}")
        self.cursor.execute(f"drop table if exists {table_name_2}")
        self.cursor.execute(f"select rows,cols from {self.vars[1]}")
        shape_1 = self.cursor.fetch()
        self.cursor.execute(f"select rows,cols from {self.vars[2]}")
        shape_2 = self.cursor.fetch()
        self.cursor.execute(f"select data from {self.vars[1]}")
        data_1 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select data from {self.vars[2]}")
        data_2 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"create table {table_name_1}(rows int,cols int,trans int,data double precision[])")
        self.cursor.execute(f"create table {table_name_2}(rows int,cols int,trans int,data double precision[])")
        if grad_output == 1:
        # 求self.vars[1]的导数,对self.vars[2]依次行求和
            sum_data_1 = []
            for i in range(shape_2[0][0]):
                total = 0
                for j in range(shape_2[0][1]):
                    total += data_2[i * shape_2[0][1] + j]
                sum_data_1.append(total)
            new_data_1 = []
            for i in range(shape_1[0][0]):
                for j in range(shape_1[0][1]):
                    new_data_1.append(sum_data_1[j])
            self.cursor.execute(f"insert into {table_name_1} values({shape_1[0][0]},{shape_1[0][1]},0,array{new_data_1})")
            # 求self.vars[2]的导数,对self.vars[1]依次列求和
            sum_data_2 = []
            for i in range(shape_1[0][1]):
                total = 0
                for j in range(shape_1[0][0]):
                    total += data_1[i + j * shape_1[0][1]]
                sum_data_2.append(total)
            new_data_2 = []
            for i in range(shape_2[0][0]):
                for j in range(shape_2[0][1]):
                    new_data_2.append(sum_data_2[i])
            self.cursor.execute(f"insert into {table_name_2} values({shape_2[0][0]},{shape_2[0][1]},0,array{new_data_2})")
        else:
            # matmul:(a,b)*(b,c)=(a,c)
            self.cursor.execute(f"select rows,cols from {grad_output}")
            shape_0 = self.cursor.fetch()
            self.cursor.execute(f"select data from {grad_output}")
            data_0 = str_to_list(self.cursor.fetch()[0][0])
            new_data_1 = []
            for i in range(shape_0[0][0]):
                for j in range(shape_2[0][0]):
                    sum = 0
                    try:
                        for k in range(shape_0[0][1]):
                            sum += data_0[i * shape_0[0][1] + k] * data_2[j * shape_2[0][1] + k]
                    except IndexError:
                        print(sum)
                    new_data_1.append(sum)
            self.cursor.execute(f"insert into {table_name_1} values({shape_1[0][0]},{shape_1[0][1]},0,array{new_data_1})")
            new_data_2 = []
            for i in range(shape_1[0][1]):
                for j in range(shape_0[0][1]):
                    sum = 0
                    for k in range(shape_1[0][0]):
                        sum += data_1[k * shape_1[0][1] + i] * data_0[k * shape_2[0][1] + j]
                    new_data_2.append(sum)
            self.cursor.execute(f"insert into {table_name_2} values({shape_2[0][0]},{shape_2[0][1]},0,array{new_data_2})")
        return table_name_1, table_name_2


class DOT(Node):
    def __init__(self, **kwargs):
        super().__init__(20, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_dot('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        # self.conn.commit()


class INNER(Node):
    def __init__(self, **kwargs):
        super().__init__(21, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        raise Exception('暂不支持inner')


class OUTER(Node):
    def __init__(self, **kwargs):
        super().__init__(22, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        raise Exception('暂不支持outer')


class TENSORDOT(Node):
    def __init__(self, **kwargs):
        super().__init__(23, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_tensordot('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        # self.conn.commit()


class KRON(Node):
    def __init__(self, **kwargs):
        super().__init__(24, **kwargs)


class CHOLESKY(Node):
    def __init__(self, **kwargs):
        super().__init__(25, **kwargs)


class QR(Node):
    def __init__(self, **kwargs):
        super().__init__(26, **kwargs)
        self.mode = ''

    def set_mode(self, mode):
        self.mode = mode


class SVD(Node):
    def __init__(self, **kwargs):
        super().__init__(27, **kwargs)
        self.full_matrices = True
        self.compute_uv = True
        self.hermitian = False

    def set_param(self, full_matrices, compute_uv, hermitian):
        self.full_matrices = bool(full_matrices)
        self.compute_uv = bool(compute_uv)
        self.hermitian = bool(hermitian)

    @preprocessing
    def run(self, **kwargs):
        # TODO
        raise Exception('暂不支持SVD')


class NORM(Node):
    def __init__(self, **kwargs):
        super().__init__(28, **kwargs)
        self.parameter_dict = {'ord': None, 'axis': None, 'keepdims': 0}

    def set_param(self, ord, axis, keepdims):
        self.parameter_dict['ord'] = ord
        self.parameter_dict['axis'] = axis
        self.parameter_dict['keepdims'] = keepdims


class COND(Node):
    def __init__(self, **kwargs):
        super().__init__(29, **kwargs)
        self.parameter_dict = {'p': None}

    def set_param(self, p):
        self.parameter_dict['p'] = p


class DET(Node):
    def __init__(self, **kwargs):
        super().__init__(30, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # TODO
        raise Exception('暂不支持DET')


class RANK(Node):
    def __init__(self, **kwargs):
        super().__init__(31, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        raise Exception('暂不支持rank')


class TRACE(Node):
    def __init__(self, **kwargs):
        super().__init__(32, **kwargs)
        self.parameter_dict = {'offset': 0, 'axis1': 0, 'axis2': 1, 'dtype': None, 'out': None}

    def set_param(self, offset, axis1, axis2, dtype, out):
        self.parameter_dict['offset'] = offset
        self.parameter_dict['axis1'] = axis1
        self.parameter_dict['axis2'] = axis2
        self.parameter_dict['dtype'] = dtype
        self.parameter_dict['out'] = out

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_trace('{self.vars[1]}', '{self.vars[0]}');")


class RESHAPE(Node):
    def __init__(self, **kwargs):
        super().__init__(33, **kwargs)
        self.new_shape = None
        self.order = 'C'

    def set_param(self, newshape, order):
        self.new_shape = newshape
        self.order = order

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(
            f"select db4ai_reshape('{self.vars[1]}', {self.new_shape[0]}, {self.new_shape[1]}, '{self.vars[0]}');")


class TRANSPOSE(Node):
    def __init__(self, **kwargs):
        super().__init__(34, **kwargs)


class STACK(Node):
    def __init__(self, **kwargs):
        super().__init__(35, **kwargs)
        self.axis = 0

    def set_axis(self, axis):
        self.axis = axis


class GRADIENT(Node):
    def __init__(self, **kwargs):
        super().__init__(36, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"drop table if exists {self.vars[0]};")
        self.cursor.execute(f"select * into {self.vars[0]} from {'grad_' + self.vars[1]};")


class SHAPE(Node):
    def __init__(self, **kwargs):
        super().__init__(37, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_shape('{self.vars[1]}', '{self.vars[0]}');")


class EXP(Node):
    def __init__(self, **kwargs):
        super().__init__(38, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_exp('{self.vars[1]}', '{self.vars[0]}');")


    '''
        此处套用self.vars[0]重构即可，无需做反向算子
    '''
    def backward(self, grad_output=1):
        table_name = 'grad_' + str(self.id)
        temp_table_name = 'grad_' + str(self.id) + '_temp1'
        if grad_output == 1:
            s = table_name
        else:
            s = temp_table_name
        self.cursor.execute(f"drop table if exists {s}")
        self.cursor.execute(f"select * into {s} from {self.vars[0]}")
        if grad_output != 1:
            self.op_broadcast("mul", s, grad_output, table_name)
        return table_name


# 该类为列表切片、索引，self.name为列表名，self.slice_info为切片信息
class Slice(Node):
    def __init__(self, **kwargs):
        super().__init__(39, **kwargs)
        self.slice_info = None
        self.slice_index = None

    def set_slice(self, slice_info):
        self.slice_info = slice_info
        total_slice = parse_slice(slice_info)
        self.slice_index = total_slice

    @preprocessing
    def run(self, **kwargs):
        s = copy(self.slice_index)
        self.cursor.execute(f"select rows,cols from {self.vars[1]};")
        shape = self.cursor.fetch()
        for idx in range(len(s)):
            if isinstance(s[idx], int):
                start = s[idx]
                stop = s[idx]
                s[idx] = [start, stop]
            elif isinstance(s[idx], str):
                self.cursor.execute(f"select data from {s[idx]};")
                temp = str_to_list(self.cursor.fetch()[0][0])
                s[idx] = [temp[0], temp[0]]
            elif isinstance(s[idx], slice):
                start = s[idx].start
                stop = s[idx].stop
                if isinstance(s[idx].start, str):
                    self.cursor.execute(f"select data from {s[idx].start};")
                    start = str_to_list(self.cursor.fetch()[0][0])[0]
                if s[idx].start is None:
                    # start = shape[0][0]
                    start = 0
                if isinstance(s[idx].stop, str):
                    self.cursor.execute(f"select data from {s[idx].stop};")
                    stop = str_to_list(self.cursor.fetch()[0][0])[0]
                if s[idx].stop is None:
                    # stop = shape[0][1]
                    stop = shape[0][idx]
                s[idx] = [start, stop]
        if len(s) == 1:
            if shape[0][0] == 1:
                s.insert(0, [0, 0])
            elif shape[0][1] == 1:
                s.append([0, 0])
            else:
                s.append([0, shape[0][1]])
        self.cursor.execute(
            f"select db4ai_slice('{self.vars[1]}', {s[0][0]}, {s[0][1]}, {s[1][0]}, {s[1][1]}, '{self.vars[0]}');")
        # self.conn.commit()


# 该类用来存储参数变量，如x，y
class Var(Node):
    def __init__(self, **kwargs):
        super().__init__(40, **kwargs)
        if 'vars' in kwargs.keys():
            self.set_vars(kwargs['vars'])
        self.grad_fn = None

    @preprocessing
    def run(self, **kwargs):
        if self.executor.backward_end != 0 and self.id > self.executor.backward_end:
            flag = 0
        else:
            flag = 1
        if self.vars[0] not in self.executor.tensor_dict and flag == 1:
            self.executor.tensor_dict[self.vars[0]] = Tensor(self.vars[0], self.cursor)
            self.executor.tensor_dict[self.vars[0]].set_grad_fn(self)


# 该类实例含义为当前位置值未知，占空，之后被其他类实例取代
class Blank(Node):
    def __init__(self, **kwargs):
        super().__init__(41, **kwargs)


class Deepcopy(Node):
    def __init__(self, **kwargs):
        super().__init__(42, **kwargs)


class Shallowcopy(Node):
    def __init__(self, **kwargs):
        super().__init__(43, **kwargs)


class Argmax(Node):
    def __init__(self, **kwargs):
        super().__init__(44, **kwargs)
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_argmax('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")

    def set_axis(self, axis):
        self.axis = axis


class Argmin(Node):
    def __init__(self, **kwargs):
        super().__init__(45, **kwargs)
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_argmin('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")

    def set_axis(self, axis):
        self.axis = axis


class Sign(Node):
    def __init__(self, **kwargs):
        super().__init__(46, **kwargs)


class SaveTable(Node):
    def __init__(self, **kwargs):
        super().__init__(47, **kwargs)
        self.table_name = None

    def set_name(self, table_name):
        self.table_name = table_name

    def get_name(self):
        return self.table_name


class Full(Node):
    def __init__(self, data_shape, var, num, **kwargs):
        super().__init__(48, **kwargs)
        if data_shape is None:
            self.data_shape = None
        else:
            self.data_shape = data_shape
        self.data_shape_var = {}
        # TODO: infer data_shape
        self.set_vars(var)
        self.num = eval(num)

    @preprocessing
    def run(self, **kwargs):
        if isinstance(self.data_shape, str):
            # 运行时使用变量的值填充变量名
            for name in self.data_shape_var.keys():
                self.data_shape_var[name] = self.cursor.execute(f"select data from {name}")
            # 转换
            self.data_shape = eval(self.data_shape, self.data_shape_var)
        self.cursor.execute(
            f"select db4ai_full({self.data_shape[0]}, {self.data_shape[1]}, {self.num}, '{self.vars[0]}');")

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape

    def handle_include_var(self, **change_info):
        if change_info['d_has_var']:
            self.data_shape_var = change_info['d_var']
        else:
            self.data_shape = eval(self.data_shape)


class Ones(Node):
    def __init__(self, data_shape, var, **kwargs):
        super().__init__(48, **kwargs)
        if data_shape is None:
            self.data_shape = None
        else:
            self.data_shape = data_shape
        # TODO: infer data_shape
        self.set_vars(var)
        self.data_shape_var = {}

    @preprocessing
    def run(self, **kwargs):
        if isinstance(self.data_shape, str):
            # 运行时使用变量的值填充变量名
            for name in self.data_shape_var.keys():
                self.cursor.execute(f"select data from {name};")
                self.data_shape_var[name] = str_to_list(self.cursor.fetch()[0][0])[0]
                # 转换
            self.data_shape = eval(self.data_shape, self.data_shape_var)
        self.cursor.execute(f"select db4ai_ones({self.data_shape[0]}, {self.data_shape[1]}, '{self.vars[0]}');")

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape

    def handle_include_var(self, **change_info):
        if change_info['d_has_var']:
            self.data_shape_var = change_info['d_var']
        else:
            self.data_shape = eval(self.data_shape)


class Zeros(Node):
    def __init__(self, data_shape, var, **kwargs):
        super().__init__(49, **kwargs)
        if data_shape is None:
            self.data_shape = None
        else:
            self.data_shape = data_shape
        # TODO: infer data_shape
        self.set_vars(var)
        self.data_shape_var = {}

    @preprocessing
    def run(self, **kwargs):
        if isinstance(self.data_shape, str):
            # 运行时使用变量的值填充变量名
            for name in self.data_shape_var.keys():
                self.cursor.execute(f"select data from {name};")
                self.data_shape_var[name] = str_to_list(self.cursor.fetch()[0][0])[0]
                # 转换
            self.data_shape = eval(self.data_shape, self.data_shape_var)
        self.cursor.execute(f"drop table if exists {self.vars[0]}")
        self.cursor.execute(f"select db4ai_zeros({self.data_shape[0]}, {self.data_shape[1]}, '{self.vars[0]}');")
        # self.conn.commit()

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape

    def handle_include_var(self, **change_info):
        if change_info['d_has_var']:
            self.data_shape_var = change_info['d_var']
        else:
            self.data_shape = eval(self.data_shape)


class SUM(Node):
    def __init__(self, **kwargs):
        super().__init__(50, **kwargs)
        self.axis = 0

    def set_axis(self, axis):
        self.axis = axis

    '''
         db4ai_sum将输入表，按列求和（输入参数==0）或按行求和（输入参数==1）,结果保存在输出表中。
    '''

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_sum('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")
        # self.conn.commit()

    def backward(self, grad_output=1):
        return grad_output


class Relu(Node):
    def __init__(self, **kwargs):
        super().__init__(51, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"drop table if exists {self.vars[0]}")
        self.cursor.execute(f"select * into {self.vars[0]} from {self.vars[1]};")
        self.cursor.execute(f"select data from {self.vars[1]};")
        data = str_to_list(self.cursor.fetch()[0][0])
        for i in range(len(data)):
            if data[i] < 0:
                data[i] = 0
        self.cursor.execute(f"update {self.vars[0]} set data = array{data}")

    def backward(self, grad_output=1):
        table_name = 'grad_' + str(self.id)
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        if grad_output == 1:
            s = table_name
        if grad_output != 1:
            s = table_name_temp1
        self.cursor.execute(f"drop table if exists {s}")
        self.cursor.execute(f"select * into {s} from {self.vars[0]};")
        self.cursor.execute(f"select data from {self.vars[0]};")
        data = str_to_list(self.cursor.fetch()[0][0])
        for i in range(len(data)):
            if data[i] > 0:
                data[i] = 1
        self.cursor.execute(f"update {self.vars[0]} set data = array{data}")
        if grad_output != 1:
            self.op_broadcast("mul", s, grad_output,table_name)
        return table_name


class Tanh(Node):
    def __init__(self, **kwargs):
        super().__init__(52, **kwargs)

    '''参数待补充'''

    def backward(self, grad_output):
        input_x, = self.saved_tensors
        grad_x = grad_output * (1 - torch.pow(
            ((torch.exp(input_x) - torch.exp(-1 * input_x)) / (torch.exp(input_x) + torch.exp(-1 * input_x))), 2))
        return grad_x


class Softmax(Node):
    def __init__(self, **kwargs):
        super().__init__(53, **kwargs)
        self.dim = 1

    def set_dim(self, dim):
        self.dim = dim

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_softmax('{self.vars[1]}', {self.dim}, '{self.vars[0]}');")

    '''
        softmax求导较为特殊，设输入为m*n矩阵a，则输出也为m*n矩阵b，求导时，a中每个元素a(i,j)求导结果为m*n矩阵c(a(i,j)),c中除第i行外均为全0
        根据前向传播，grad_output也应为m*n矩阵g,则Σg(i,j)*c(a(i,j))为最终结果(乘以上游梯度后的最终梯度)
        由于m.n较大时建表过多可能影响性能，此处不再将算子梯度与前向梯度区分开
    '''
    def backward(self, grad_output):
        table_name = 'grad_' + str(self.id)
        assert grad_output != 1
        self.cursor.execute(f"drop table if exists {table_name}")
        self.cursor.execute(f"select * into {table_name} from {self.vars[0]}")

        self.cursor.execute(f"select rows,cols from {grad_output};")
        grad_output_shape = self.cursor.fetch()
        self.cursor.execute(f"select data from {grad_output};")
        grad_output_data = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select rows,cols from {self.vars[0]};")
        softmax_shape = self.cursor.fetch()
        assert softmax_shape == grad_output_shape
        self.cursor.execute(f"select data from {self.vars[0]};")
        softmax_data = str_to_list(self.cursor.fetch()[0][0])

        new_data = []
        for i in range(softmax_shape[0][0]):
            for r in range(softmax_shape[0][1]):
                new_data.append(0)
            for j in range(softmax_shape[0][1]):
                for k in range(softmax_shape[0][1]):
                    index = i * softmax_shape[0][1] + k
                    if j == k:
                        new_data[index] += softmax_data[index] * (1 - softmax_data[index]) * grad_output_data[i * grad_output_shape[0][1] + j]
                    else:
                        new_data[index] += -softmax_data[i * softmax_shape[0][1] + j] * softmax_data[index] * grad_output_data[i * grad_output_shape[0][1] + j]

        self.cursor.execute(f"update {table_name} set data = array{new_data};")
        return table_name


class Sigmod(Node):
    def __init__(self, **kwargs):
        super().__init__(54, **kwargs)


class Elu(Node):
    def __init__(self, **kwargs):
        super().__init__(55, **kwargs)
        self.alpha = None

    def set_alpha(self, alpha):
        self.alpha = eval(alpha)


class Adam(Node):
    def __init__(self, **kwargs):
        super().__init__(56, **kwargs)
        self.learning_rate = None

    def set_learning_rate(self, learning_rate):
        self.learning_rate = eval(learning_rate)


class MEAN(Node):
    def __init__(self, **kwargs):
        super().__init__(57, **kwargs)
        self.axis = 2

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_mean('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")
        # self.conn.commit()

    def set_axis(self, axis):
        self.axis = axis

    '''
        求导结果为各维度长度乘积的倒数
    '''
    def backward(self, grad_output=1):
        table_name = "grad_" + str(self.id)
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        if grad_output == 1:
            s = table_name
        else:
            s = table_name_temp1
        self.cursor.execute(f"select rows,cols from {self.vars[1]}")
        shape = self.cursor.fetch()[0]
        if self.axis == 0:
            val = 1 / shape[1]
        elif self.axis == 1:
            val = 1 / shape[0]
        else:
            val = 1 / (shape[0] * shape[1])
        data = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                data.append(val)
        self.cursor.execute(f"drop table if exists {s}")
        self.cursor.execute(f"select * into {s} from {self.vars[1]}")
        self.cursor.execute(f"update {s} set data = array{data}")
        if grad_output != 1:
            self.op_broadcast("mul", s, grad_output, table_name)
        return table_name


class MAX(Node):
    def __init__(self, **kwargs):
        super().__init__(58, **kwargs)
        self.axis = 2

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_max('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")

    def set_axis(self, axis):
        self.axis = axis


class MIN(Node):
    def __init__(self, **kwargs):
        super().__init__(59, **kwargs)
        self.axis = 2

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_min('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")

    def set_axis(self, axis):
        self.axis = axis


class Abs(Node):
    def __init__(self, **kwargs):
        super().__init__(60, **kwargs)
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_abs('{self.vars[1]}', '{self.vars[0]}');")

    def set_axis(self, axis):
        self.axis = axis


class SplitDataset(Node):
    def __init__(self, **kwargs):
        # SplitDataset(data:variable, size:tensor)
        super().__init__(61, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # TODO：代码实现@樊宣伯, @路亚彬
        raise Exception('SplitDataset未实现')


class Parameters(Node):
    def __init__(self, **kwargs):
        # Parameters(set_name, var1, var2, ......)
        # 用来声明参数并对参数打包
        super().__init__(62, **kwargs)
        # TODO: 解析参数集合的名字, 把vari存入self.vars[i]
        self.set_name = None

    @preprocessing
    def run(self, **kwargs):
        self.executor.parameters[self.set_name] = self.vars[1:]


class SaveParameters(Node):
    def __init__(self, **kwargs):
        # SaveParameters(set_name, path)
        # 用来保存指定名称的参数集
        super().__init__(63, **kwargs)
        # TODO: 解析参数集合的名字和存储路径
        self.set_name = None
        self.path = None

    @preprocessing
    def run(self, **kwargs):
        # paras = self.executor.parameters[self.set_name]
        # tmp = {}
        # for para in paras:
        #     tmp[para] = self.executor.var_dict[para]
        # dump_tensor(tmp, self.path)
        # TODO: @张恺欣， @路亚彬
        raise Exception('SaveParameters暂未实现')


class LoadParameters(Node):
    def __init__(self, **kwargs):
        # LoadParameters(set_name, path)
        # 用来保存指定名称的参数集
        super().__init__(64, **kwargs)
        # TODO: 解析参数集合的名字和存储路径
        self.set_name = None
        self.path = None

    @preprocessing
    def run(self, **kwargs):
        # paras = self.executor.parameters[self.set_name]
        # tensors = load_tensor(self.path)
        # for para in paras:
        #     self.executor.var_dict[para] = tensors[para]
        # TODO: @张恺欣， @路亚彬
        raise Exception('SaveParameters暂未实现')


class AUC(Node):
    def __init__(self, **kwargs):
        super().__init__(65, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select data from {self.vars[1]}")
        test_y = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select data from {self.vars[2]}")
        pred = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"drop table if exists {self.vars[0]}")
        self.cursor.execute(f"create table {self.vars[0]} (rows int,cols int,trans int,data double precision[])")
        try:
            if len(torch.unique(torch.tensor(test_y))) > 2:
                self.cursor.execute(
                    f"insert into {self.vars[0]} values (1,1,0,array{[sk_metrics.roc_auc_score(test_y, pred, multi_class='ovr')]})")
            else:
                self.cursor.execute(
                    f"insert into {self.vars[0]} values (1,1,0,array{[sk_metrics.roc_auc_score(test_y, pred)]})")
        except ValueError:
            print(str(self.id) + ': ValueError')


class MSE(Node):
    def __init__(self, **kwargs):
        super().__init__(66, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select data from {self.vars[1]}")
        data_1 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"select data from {self.vars[2]}")
        data_2 = str_to_list(self.cursor.fetch()[0][0])
        self.cursor.execute(f"drop table if exists {self.vars[0]}")
        self.cursor.execute(f"create table {self.vars[0]}(rows int,cols int,trans int,data double precision[])")
        self.cursor.execute(
            f"insert into {self.vars[0]} values (1,1,0,array{[sk_metrics.mean_squared_error(data_1, data_2)]})")


class F1(Node):
    def __init__(self, **kwargs):
        super().__init__(67, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"drop table if exists {self.vars[0]};")
        self.cursor.execute(f"select db4ai_f1('{self.vars[1]}','{self.vars[2]}','{self.vars[0]}');")


class REVERSE(Node):
    def __init__(self, **kwargs):
        super().__init__(68, **kwargs)
        # TODO: dims

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_reverse('{self.vars[1]}', 1, '{self.vars[0]}');")


class ARGSORT(Node):
    def __init__(self, **kwargs):
        super().__init__(69, **kwargs)
        self.dim = 2

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_argsort('{self.vars[1]}', {self.dim}, '{self.vars[0]}');")

    def set_axis(self, dim):
        self.dim = dim


class SORT(Node):
    def __init__(self, **kwargs):
        super().__init__(70, **kwargs)
        self.dim = 2

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_sort('{self.vars[1]}', {self.dim}, '{self.vars[0]}');")

    def set_dim(self, dim):
        self.dim = dim


class ACC(Node):
    def __init__(self, **kwargs):
        super().__init__(71, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"drop table if exists {self.vars[0]}")
        self.cursor.execute(f"select db4ai_acc('{self.vars[1]}', '{self.vars[2]}', 1, '{self.vars[0]}')")


class RECALL(Node):
    def __init__(self, **kwargs):
        super().__init__(72, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        pass
        # self.cursor.execute(f"select db4ai_recall('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")


class PRECISION(Node):
    def __init__(self, **kwargs):
        super().__init__(73, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        pass
        # self.cursor.execute(f"select db4ai_precision('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")


"""
Backward计算叶节点梯度，存放于Node_grad表中
"""


class Backward(Node):
    def __init__(self, **kwargs):
        super().__init__(74, **kwargs)
        self.retain_graph = False
        if 'retain_graph' in kwargs.keys():
            self.retain_graph = kwargs['retain_graph']

    @preprocessing
    def run(self, **kwargs):

        #
        """
        初始化
        """
        tensors = []
        drop_table_list = []
        tensors.append(self.executor.tensor_dict[self.vars[0]])
        """
        遍历所有点 
        """
        while tensors:
            tensor = tensors.pop()
            # 计算算子梯度
            if tensor.grad_fn is not None and not isinstance(tensor.grad_fn, Var):
                q = "tensor: " + str(tensor.grad_fn.id)
                print(q)
                node = tensor.grad_fn
                table_name = 'grad_input_' + str(node.id)
                self.cursor.execute(f"select count(*) from pg_class where relname = '{table_name}';")
                node_flag = self.cursor.fetch()[0][0] == 1
                if node_flag:
                    tup = node.backward(table_name)
                else:
                    tup = node.backward()
                if not isinstance(tup, tuple):
                    tup = (tup,)
                # id由小到大排序父节点
                fathers = []

                for pre_tensor in tensor.get_next():
                    if pre_tensor.grad_fn is not None:
                        index = 0
                        for father in fathers:
                            if pre_tensor.grad_fn.id > father.id:
                                index += 1
                        fathers.insert(index, pre_tensor.grad_fn)
                    tensors.append(pre_tensor)
                # 对父节点中Vars添加到fathers中
                for pre_fa in node.fathers:
                    if self.executor.var_dict[pre_fa.vars[0]] not in fathers and pre_fa.__class__.__name__ in ['Var', 'Val']:
                        flag = 1
                        for father in fathers:
                            if pre_fa.vars == father.vars:
                                flag = 0
                        if flag == 1:
                            flag2 = 0
                            for father in fathers:
                                true_father = father
                                if father.vars[0] not in node.vars:
                                    ff = 1
                                    temp = [father]
                                    while ff:
                                        tt = temp.pop(0)
                                        for nex in tt.next_nodes():
                                            temp.append(nex)
                                            if nex in node.pre_nodes():
                                                true_father = nex
                                                ff = 0
                                if pre_fa.id == true_father.id:
                                    flag2 = 1
                                    break
                                if pre_fa.id > true_father.id:
                                    index += 1
                            if flag2 != 1:
                                fathers.insert(index, pre_fa)
                index = 0
                for father in fathers:
                    if father.with_grad is True:
                        if father.__class__.__name__ is 'Var':
                            fa_table_name = "grad_" + father.vars[0]
                        elif father.__class__.__name__ in operators:
                            fa_table_name = "grad_input_" + str(father.id)
                            drop_table_list.append(fa_table_name)
                        else:
                            index += 1
                            continue
                        _index = index
                        while _index >= len(tup):
                            _index = _index - len(tup)
                        self.cursor.execute(f"select count(*) from pg_class where relname = '{fa_table_name}';")
                        if self.cursor.fetch()[0][0] == 0:
                            if tup[_index] == 1:
                                self.cursor.execute(f"create table {fa_table_name}(rows integer,cols integer, "
                                                    f"trans integer,data double precision[])")
                                self.cursor.execute(f"insert into {fa_table_name} values (1,1,0,array{[1.0]})")
                            else:
                                self.cursor.execute(f"select rows,cols from {father.vars[0]}")
                                shape = self.cursor.fetch()
                                self.groupby(tup[_index],fa_table_name,shape[0][0],shape[0][1])
                        else:
                            if father.__class__.__name__ is 'Var':
                                old_fa_table_name = 'old_' + fa_table_name
                                self.cursor.execute(f"drop table if exists {old_fa_table_name}")
                                self.cursor.execute(f"select * into {old_fa_table_name} from {fa_table_name}")
                                self.cursor.execute(f"select rows,cols from {fa_table_name}")
                                shape = self.cursor.fetch()
                                # self.op_broadcast("add", old_fa_table_name, tup[_index], fa_table_name)
                                group_by_table = 'group_by_'+tup[_index]
                                self.groupby(tup[_index],group_by_table,shape[0][0],shape[0][1])
                                self.op_broadcast("add", old_fa_table_name, group_by_table, fa_table_name)
                                drop_table_list.extend([old_fa_table_name,group_by_table])
                            else:
                                self.cursor.execute(f"drop table if exists {fa_table_name}")
                                self.cursor.execute(f"select * into {fa_table_name} from {tup[_index]}")
                    index += 1
        self.compare_with_torch()
        # c = Test()
        # c.test(self.cursor)
        for i in drop_table_list:
            self.cursor.execute(f"drop table if exists {i}")
        self.cursor.execute("select data from __0loss")
        print(str_to_list(self.cursor.fetch()[0][0])[0])

        if not self.retain_graph:
            for i in self.executor.tensor_dict.keys():
                self.executor.tensor_dict[i].clear_next()
            # self.conn.commit()

    def change_table_to_tensor(self,table_name):
        self.cursor.execute(f"select rows,cols from {table_name}")
        shape = self.cursor.fetch()
        self.cursor.execute(f"select data from {table_name}")
        data = str_to_list(self.cursor.fetch()[0][0])
        tensor = []
        for i in range(shape[0][0]):
            l = []
            for j in range(shape[0][1]):
                l.append(float(data[i*shape[0][1] + j]))
            tensor.append(l)
        tensor = torch.tensor(tensor,requires_grad=True)
        return tensor

    def compare_with_torch(self):
        x = self.change_table_to_tensor("x")
        y = self.change_table_to_tensor("y")
        w_0 = self.change_table_to_tensor("__0w_0")
        b_0 = self.change_table_to_tensor("__0b_0")
        w_1 = self.change_table_to_tensor("__0w_1")
        b_1 = self.change_table_to_tensor("__0b_1")
        w_2 = self.change_table_to_tensor("__0w_2")
        b_2 = self.change_table_to_tensor("__0b_2")
        w_3 = self.change_table_to_tensor("__0w_3")
        b_3 = self.change_table_to_tensor("__0b_3")
        output_0 = torch.relu(torch.matmul(x,w_0)+b_0)
        output_1 = torch.relu(torch.matmul(output_0,w_1)+b_1)
        output_2 = torch.relu(torch.matmul(output_1,w_2)+b_2)
        output_3 = 1/(1+torch.exp(-1*(torch.matmul(output_2,w_3)+b_3)))
        loss = -1 * torch.mean(y*torch.log(output_3)+(1-y)*torch.log(1-output_3))
        loss.backward()
        print(w_0.grad)
        print(w_1.grad)
        print(w_2.grad)
        print(w_3.grad)
        print(b_0.grad)
        print(b_1.grad)
        print(b_2.grad)
        print(b_3.grad)






class WLS(Node):
    def __init__(self, **kwargs):
        super().__init__(75, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.tensor(sklearn.linear_model.LinearRegression(
        #     self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]],
        #     sample_weight=self.executor.var_dict[self.vars[2]]))
        raise Exception('WLS暂不支持')


class REPEAT(Node):
    def __init__(self, **kwargs):
        super().__init__(76, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(
            f"select db4ai_repeat('{self.vars[1]}', '{self.vars[4]}', '{self.vars[3]}', {self.vars[0]}');")

    '''
        求导结果为扩充倍数乘积
    '''
    def backward(self, grad_output=1):
        table_name = "grad_" + str(self.id)
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        if grad_output == 1:
            s = table_name
        else:
            s = table_name_temp1
        self.cursor.execute(f"select data from {self.vars[2]}")
        vars_2 = str_to_list(self.cursor.fetch()[0][0])[0]
        self.cursor.execute(f"select data from {self.vars[3]}")
        vars_3 = str_to_list(self.cursor.fetch()[0][0])[0]
        self.cursor.execute(f"select data from {self.vars[4]}")
        vars_4 = str_to_list(self.cursor.fetch()[0][0])[0]
        t = vars_2 * vars_3 * vars_4
        self.cursor.execute(f"select rows,cols from {self.vars[1]}")
        shape = self.cursor.fetch()
        data = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                data.append(t)
        self.cursor.execute(f"select * into {s} from {self.vars[1]}")
        self.cursor.execute(f"update {s} set data = {data}")
        # self.conn.commit()
        if grad_output == 1:
            self.op_broadcast("mul", s, grad_output, table_name)
        return table_name


class UNSQUEEZE(Node):
    def __init__(self, **kwargs):
        super().__init__(77, **kwargs)
        self.dim = None

    @preprocessing
    def run(self, **kwargs):
        raise Exception('暂不支持UNSQUEEZE')

    def set_dim(self, dim):
        self.dim = int(dim)


class CleanGrad(Node):
    def __init__(self, **kwargs):
        super().__init__(78, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"drop table if exists {'grad_' + self.vars[0]}")
        # self.conn.commit()


class Negative(Node):
    def __init__(self, **kwargs):
        super().__init__(79, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"drop table if exists {self.vars[0]}")
        self.cursor.execute(f"select * into {self.vars[0]} from {self.vars[1]}")
        self.cursor.execute(f"select data from {self.vars[1]}")
        data = str_to_list(self.cursor.fetch()[0][0])
        update = []
        for i in data:
            update.append(-1 * i)
        self.cursor.execute(f"update {self.vars[0]} set data = array{update}")
        # self.conn.commit()

    '''
        求导值为-1，直接存储即可
    '''
    def backward(self, grad_output=1):
        table_name = 'grad_' + str(self.id)
        table_name_temp1 = 'grad_' + str(self.id) + '_temp1'
        if grad_output == 1:
            s = table_name
        if grad_output != 1:
            s = table_name_temp1
        self.cursor.execute(f"create table {s}(rows int, cols int,trans int,data double precision[])")
        self.cursor.execute(f"insert into {s} values (1,1,0,array{[-1]})")
        if grad_output != 1:
            self.op_broadcast("mul", s, grad_output, table_name)
            # self.conn.commit()
        return table_name


def shallow_copy(fun):
    @wraps(fun)
    def decorated(*args, **kwargs):
        list_args = []
        for para in args:
            if isinstance(para, list):
                list_args.append(copy(para))
            else:
                list_args.append(para)
        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = copy(value)
        return fun(*list_args, **kwargs)

    return decorated


# 通过globals方法，以类名选择类进行实例化
@shallow_copy
def InstantiationClass(nodeId, nodeType, branches=None, with_grad=False, **otherField):
    if nodeType == 'CreateTensor' or nodeType == 'Zeros' or nodeType == 'Ones':
        data_shape = otherField['data_shape']
        var = otherField['var']
        node = globals()[nodeType](data_shape, var, id=nodeId, branches=branches, with_grad=with_grad)
    elif nodeType == 'Full':
        data_shape = otherField['data_shape']
        var = otherField['var']
        num = otherField['num']
        node = globals()[nodeType](data_shape, var, num=num, id=nodeId, branches=branches, with_grad=with_grad)
    elif nodeType == 'Sql':
        t_info = otherField['t_info']
        var = otherField['var']
        node = globals()[nodeType](t_info, var, id=nodeId, branches=branches, with_grad=with_grad)
    elif nodeType == 'Random':
        boundary = otherField['boundary']
        data_shape = otherField['data_shape']
        type = otherField['type']
        var = otherField['var']
        node = globals()[nodeType](boundary, data_shape, type, var, id=nodeId, branches=branches, with_grad=with_grad)
    elif nodeType == 'Val':
        val = otherField['val']
        if 'var' in otherField.keys():
            var = otherField['var']
            node = globals()[nodeType](var, id=nodeId, branches=branches, with_grad=with_grad, val=val)
        else:
            node = globals()[nodeType](var=[], id=nodeId, branches=branches, with_grad=with_grad, val=val)
    elif nodeType == 'Assignment':
        var_li = otherField['var_li']
        node = globals()[nodeType](var_li, id=nodeId, branches=branches, with_grad=with_grad)
    elif nodeType == 'Loop':
        condition = otherField['condition']
        loop_id = otherField['loop_id']
        node = globals()[nodeType](condition, loop_id, id=nodeId, branches=branches, with_grad=with_grad)
    elif nodeType == 'LoopEnd' or nodeType == 'Break':
        loop_id = otherField['loop_id']
        node = globals()[nodeType](loop_id, id=nodeId, branches=branches, with_grad=with_grad)
    else:
        node = globals()[nodeType](id=nodeId, branches=branches, with_grad=with_grad, **otherField)
    return node


