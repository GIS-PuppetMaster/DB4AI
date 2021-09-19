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
from sklearn import metrics as sk_metrics
import pickle as pk
from Analyze_expression import all_operator


def preprocessing(fun):
    @wraps(fun)
    def decorated(node, **kwargs):
        # todo 自动类型转换
        '''for input in node.vars[1:]:
            if node.physic_algorithm == 'madlib' and not isinstance(node.executor.var_dict[input], str):
                # sql->torch
                pass
            elif node.physic_algorithm != 'madlib' and isinstance(node.executor.var_dict[input], str):
                # torch->sql
                pass'''
        if not node.with_grad and not isinstance(node, GRADIENT):
            with torch.no_grad():
                return fun(node, **kwargs)
        else:
            return fun(node, **kwargs)

    return decorated


def dump_tensor(tensors: dict, path: str):
    with open(path, 'wb') as f:
        pk.dump(tensors, f)


def fill_slice_var(slice_index, executor):
    s = copy(slice_index)
    for idx in range(len(s)):
        if isinstance(s[idx], str):
            s[idx] = int(executor.var_dict[s[idx]])
        elif isinstance(s[idx], slice):
            start = s[idx].start
            step = s[idx].step
            stop = s[idx].stop
            if isinstance(s[idx].start, str):
                start = int(executor.var_dict[s[idx].start])
            if isinstance(s[idx].step, str):
                step = int(executor.var_dict[s[idx].step])
            if isinstance(s[idx].stop, str):
                stop = int(executor.var_dict[s[idx].stop])
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
        self.conn = psycopg2.connect(database="postgres", user="postgres", host="114.115.156.203", port="2333")
        self.cursor = self.conn.cursor()

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
    def set_conn(self):
        self.cursor = self.conn.cursor()

    def __repr__(self):
        return f'id:{self.id}, branches:{self.branches}, vars:{self.vars}'

    def sql_select_grad(self, id):
        table_name = "grad_" + str(id)
        self.cursor.execute(f"select count(*) from pg_class where relname = '{table_name}'")
        flag = self.cursor.fetchall()[0][0] == 1
        if flag:
            grad = []
            self.cursor.execute(f"select val from {table_name}")
            rows = self.cursor.fetchall()
            if len(rows) == 0:
                grad = rows[0]
            else:
                for row in rows:
                    grad.append(row[0])
        else:
            grad = 1
        return grad

    def sql_create_grad(self, id, grad):
        table_name = "grad_" + str(id)
        self.cursor.execute(f"DROP TABLE IF EXISTS " + {table_name})
        if isinstance(grad, list):
            if isinstance(grad[0], list):
                self.cursor.execute(f"create table {table_name}(row serial, val numeric[][])")
            else:
                self.cursor.execute(f"create table {table_name}(row serial, val numeric[])")
        else:
            self.cursor.execute(f"create table {table_name}(row serial, val numeric)")
        self.cursor.execute(f"insert into {table_name}(val) values('{grad}')")
        self.conn.commit()

    def sql_update(self, table_name, var_name, value):
        if isinstance(value, list):
            if isinstance(value[0], list):
                self.cursor.execute(f"update {table_name} set val_2 = {value} where name = '{var_name}'")
            else:
                self.cursor.execute(f"update {table_name} set val_2 = {value} where name = '{var_name}'")
        else:
            self.cursor.execute(f"update {table_name} set val_2 = {value} where name = {var_name}'")


# 通过继承实现的其它节点类
class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)


# 创建张量所用节点
class CreateTensor(Node):
    def __init__(self, data_shape, var, **kwargs):
        super().__init__(1, **kwargs)
        # if isinstance(data_shape, tuple):
        #     self.data_shape = data_shape
        # elif isinstance(data_shape, str):
        #     self.data_shape = eval(data_shape)
        # elif data_shape is None:
        #     self.data_shape = None
        # TODO: infer data_shape
        self.set_vars(var)
        self.grad = None

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = None


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
        '''if self.vars[0] not in self.executor.var_dict:
            self.executor.var_dict[self.vars[0]] = torch.tensor(self.value)'''
        self.cursor.execute(f"DROP TABLE IF EXISTS {self.vars[0]}")
        self.cursor.execute(f"create table {self.vars[0]}(row integer primary key,val double precision)")
        self.cursor.execute(f"insert into {self.vars[0]}(row,val) values (1, {self.value})")
        self.conn.commit()

    def get_val(self):
        return self.value

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = (1,)


class Sql(Node):
    def __init__(self, t_info, var, **kwargs):
        super().__init__(3, **kwargs)
        self.t_search_sentences = t_info
        self.vars = var
        self.shape = None
        # self.batch_size = None  # TODO: 自动选择batch_size

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = None  # TODO:get data

    def infer_data(self):
        for edge in self.out_edges:
            # TODO: 使用SQL查询该表的维度
            self.shape = None
            edge.data_shape = self.shape


class Random(Node):
    def __init__(self, boundary, data_shape, distribution, var, **kwargs):
        super().__init__(4, **kwargs)
        self.boundary = boundary
        self.vars = var
        self.data_shape = data_shape
        # 记录data shape和boundary中可能出现的变量名
        self.data_shape_var = {}
        self.boundary_var = {}
        if distribution == '' or distribution is None or (isinstance(distribution, list) and len(distribution) == 0):
            self.distribution = 'normal'
        else:
            self.distribution = distribution
        self.dis_args = 'mu=1000,sigma=1000'

    @preprocessing
    def run(self, **kwargs):
        # if self.physic_algorithm != 'madlib':
        #     # 如果data shape中包含var
        #     if isinstance(self.data_shape, str):
        #         # 运行时使用变量的值填充变量名
        #         for name in self.data_shape_var.keys():
        #             self.data_shape_var[name] = int(self.executor.var_dict[name])
        #         # 转换
        #         self.data_shape = eval(self.data_shape, self.data_shape_var)
        #     # 如果boundary中包含var
        #     if isinstance(self.boundary, str):
        #         # 运行时使用变量的值填充变量名
        #         for name in self.boundary_var.keys():
        #             self.boundary_var[name] = int(self.executor.var_dict[name])
        #         # 转换
        #         self.boundary = eval(self.boundary, self.boundary_var)
        #     if self.distribution == 'normal':
        #         # boundary[0]=lower_boundary, boundary[1]=upper_boundary
        #         tensor = torch.randn(self.data_shape) * (self.boundary[1] - self.boundary[0]) + self.boundary[0]
        #     elif self.distribution == 'gauss':
        #         # boundary[0]=mu, boundary[1]=sigma
        #         tensor = torch.randn() * self.boundary[1] + self.boundary[0]
        #     elif self.distribution == 'int':
        #         tensor = torch.randint(low=self.boundary[0], high=self.boundary[1], size=self.data_shape)
        #     else:
        #         raise Exception(f'Not supported distribution:{self.distribution}')
        #     if self.with_grad:
        #         tensor.requires_grad = True
        #     self.executor.var_dict[self.vars[0]] = tensor
        # else:
        #     # TODO:
        # 如果data shape中包含var
        if isinstance(self.data_shape, str):
            # 运行时使用变量的值填充变量名
            for name in self.data_shape_var.keys():
                self.data_shape_var[name] = self.cursor.execute(f"select val from {name};")
            # 转换
            self.data_shape = eval(self.data_shape, self.data_shape_var)
        self.cursor.execute(
            f"select db4ai_random({self.data_shape[0]}, {self.data_shape[1]}, {self.distribution}, {self.dis_args}, {self.vars[0]}")
        self.conn.commit()

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
            self.dead_cycle = self.executor.var_dict[self.dead_cycle]
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
                para[var_name] = self.executor.var_dict[var_node.vars[0]]
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
                    para[var_name] = self.executor.var_dict[var_node.vars[0]]
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
        self.cursor.execute(f"select count(*) from pg_class where relname = '{self.vars[1]}'")
        rows = self.cursor.fetchall()
        flag_right = rows[0][0] == 1
        if flag_right is True:
            self.cursor.execute(f"DROP TABLE IF EXISTS {self.vars[0]}")
            self.cursor.execute(f"select * into {self.vars[0]} from {self.vars[1]}")
            self.conn.commit()
        self.cursor.execute(f"select count(*) from pg_class where relname = '{self.vars[0]}'")
        rows = self.cursor.fetchall()
        flag_left = rows[0][0] == 1
        # right = self.executor.var_dict[self.vars[1]]
        # left = self.executor.var_dict[self.vars[0]]

        '''if right is None or isinstance(right, torch.Tensor):
            if left is None or isinstance(left, torch.Tensor):
                if self.slice is None and flag_right is True:
                    if self.update:
                        self.sql_update("var_dict", self.vars[0], right.data)
                        # self.executor.var_dict[self.vars[0]].data = right.data
                    else:
                        self.sql_update("var_dict", self.vars[0], right)
                        # self.executor.var_dict[self.vars[0]] = right

                else:
                    s = fill_slice_var(self.slice, self.executor)
                    # if self.vars[0] not in self.executor.var_dict:
                    #     self.executor.var_dict[self.vars[0]] = torch.empty(self.executor.var_shape[self.vars[0]])
                    if self.update:
                        self.executor.var_dict[self.vars[0]].data.__setitem__(s, right.data)
                    else:
                        self.executor.var_dict[self.vars[0]].__setitem__(s, right)
            else:
                if self.slice is None:
                    # TODO: transpose to madlib matrix then assignment
                    torch.add()
                    pass
                else:
                    s = copy(self.slice)
                    for idx in range(len(s)):
                        if isinstance(s[idx], str):
                            s[idx] = int(self.executor.var_dict[s[idx]])
                    # TODO: set_item with madlib
        # 如果右部是madlib matrix
        else:
            if left is None or isinstance(left, str):
                # TODO: 赋值给madlib matrix
                self.executor.var_dict[self.vars[0]] = self.vars[0]
            else:
                # TODO: madlib_to_tensor，赋值
                pass
        if self.with_grad and self.executor.var_dict[self.vars[0]] is not None and not self.executor.var_dict[
            self.vars[0]].requires_grad:
            self.executor.var_dict[self.vars[0]].requires_grad = True'''


class Add(Node):
    def __init__(self, **kwargs):
        super().__init__(12, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        '''if self.physic_algorithm != 'madlib':
            self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] + self.executor.var_dict[
                self.vars[2]]
        else:
            pass'''
        self.cursor.execute(f"select db4ai_add('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        self.conn.commit()

    def backward(self, grad_output=1):
        return grad_output, grad_output


class Sub(Node):
    def __init__(self, **kwargs):
        super().__init__(13, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select db4ai_sub('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        self.conn.commit()
        '''if self.physic_algorithm != 'madlib':
            self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] - self.executor.var_dict[
                self.vars[2]]
        else:
            pass'''

    def backward(self, grad_output=1):
        table_name = 'grad_' + self.id + '2'
        if grad_output != 1:
            self.cursor.execute(f"select * into {table_name} from {grad_output}")
            self.cursor.execute(f"select val from {grad_output}")
            rows = self.cursor.fetchall()
            for i in range(len(rows)):
                new_row = []
                for r in rows[i][0]:
                    r = -1 * r
                    new_row.append(r)
                self.cursor.execute(f"update {table_name} set val = {new_row} where id = {i + 1}")
            self.conn.commit()

        return grad_output, table_name


class Mul(Node):
    def __init__(self, **kwargs):
        super().__init__(14, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # if self.physic_algorithm != 'madlib':
        #     self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] * self.executor.var_dict[self.vars[2]]
        # else:
        self.cursor.execute(f"select val from {self.vars[1]}")
        rows_1 = self.cursor.fetchall()
        self.cursor.execute(f"select val from {self.vars[2]}")
        rows_2 = self.cursor.fetchall()
        if isinstance(rows_1[0][0], list) and isinstance(rows_2[0][0], list):
            if len(rows_1) == len(rows_2) and len(rows_1[0]) == len(rows_2[0]):
                self.cursor.execute(f"select db4ai_mul('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
            else:
                pass
        else:
            self.cursor.execute(f"DROP TABLE IF EXISTS {self.vars[0]}")
            self.cursor.execute(f"create table {self.vars[0]} (row integer primary key, val double precision[]);")
            if len(rows_1[0]) == 1:
                for i in range(len(rows_2)):
                    new_row = []
                    for r in rows_2[i][0]:
                        r = r * rows_1[0][0]
                        new_row.append(r)
                    self.cursor.execute(f"insert into {self.vars[0]} values({i + 1}, array{new_row});")
            elif len(rows_2[0]) == 1:
                for i in range(len(rows_1)):
                    new_row = []
                    for r in rows_1[i][0]:
                        r = r * rows_2[0][0]
                        new_row.append(r)
                    self.cursor.execute(f"insert into {self.vars[0]} values({i + 1}, array{new_row});")
        self.conn.commit()

    def backward(self, grad_output=1):
        table_name_1 = 'grad_' + str(self.id) + '_1'
        table_name_2 = 'grad_' + str(self.id) + '_2'
        self.cursor.execute(f"drop table if exists {table_name_1}")
        self.cursor.execute(f"drop table if exists {table_name_2}")
        self.cursor.execute(f"select val from {self.vars[1]}")
        rows_1 = self.cursor.fetchall()
        self.cursor.execute(f"select val from {self.vars[2]}")
        rows_2 = self.cursor.fetchall()

        if isinstance(rows_1[0][0], list):
            flag = True
        else:
            flag = False
        if flag:
            self.cursor.execute(f"select * into {table_name_1} from {self.vars[1]}")
            self.cursor.execute(f"select * into {table_name_2} from {self.vars[2]}")
            sum = 0
            for row in rows_1:
                new_row = []
                for r in row[0]:
                    sum += r
                    r = rows_2[0][0]
                    new_row.append(r)
                self.cursor.execute(f"update {table_name_1} set val = array{new_row} where row = {list(rows_1).index(row) + 1}")
            self.cursor.execute(f"update {table_name_2} set val = {sum} where row = 1")
            self.conn.commit()
        else:
            self.cursor.execute(f"select * into {table_name_1} from {self.vars[1]}")
            self.cursor.execute(f"select * into {table_name_2} from {self.vars[2]}")
            sum = 0
            for i in range(len(rows_2)):
                new_row = []
                for r in rows_2[i][0]:
                    sum += r
                    r = rows_1[0][0]
                    new_row.append(r)
                self.cursor.execute(f"update {table_name_2} set val = array{new_row} where row = {i + 1}")
            self.cursor.execute(f"update {table_name_1} set val = {sum} where row = 1")
            self.conn.commit()
        if grad_output == 1:
            return table_name_1, table_name_2
        else:
            new_table_name_1 = 'new_grad_' + str(self.id) + '_1'
            new_table_name_2 = 'new_grad_' + str(self.id) + '_2'
            self.cursor.execute(f"select * from {grad_output}")
            rows = self.cursor.fetchall()
            if isinstance(rows[0][0], list):
                self.cursor.execute(f"select db4ai.mul('{table_name_1}','{grad_output}','{new_table_name_1}'); ")
                self.cursor.execute(f"select db4ai.mul('{table_name_2}','{grad_output}','{new_table_name_2}'); ")
            else:
                if flag:
                    self.cursor.execute(f"create table {new_table_name_1}(row integer primary key, val double precision); ")

                    self.cursor.execute(f"select madlib.matrix_scalar_mult('{table_name_1}', 'row=row, val=val',"
                                    f"{rows[0][0]},'{new_table_name_1}'); ")
                    self.cursor.execute(f"select madlib.matrix_scalar_mult('{table_name_2}', 'row=row, val=val',"
                                    f"{rows[0][0]},'{new_table_name_2}'); ")
            self.conn.commit()
            return new_table_name_1, new_table_name_2


class Div(Node):
    def __init__(self, **kwargs):
        super().__init__(15, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # if self.physic_algorithm != 'madlib':
        #     self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] / self.executor.var_dict[
        #         self.vars[2]]
        # else:
        self.cursor.execute(f"select db4ai_div('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        self.conn.commit()

    def backward(self, grad_output=1):
        return 1 / self.executor.var_dict[self.vars[2]] * grad_output, -math.pow(self.executor.var_dict[self.vars[1]],
                                                                                 2) * grad_output


class LOG(Node):
    def __init__(self, **kwargs):
        super().__init__(16, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # if self.physic_algorithm != 'madlib':
        #     self.executor.var_dict[self.vars[0]] = torch.log(self.executor.var_dict[self.vars[1]])
        # else:
        self.cursor.execute(f"select db4ai_log('{self.vars[1]}', '{self.vars[0]}');")
        self.conn.commit()

    def backward(self, grad_output=1):
        temp_table_name = 'grad_' + str(self.id) + '_temp'
        self.cursor.execute(f"select val from {self.vars[1]};")
        rows = self.cursor.fetchall()
        for i in range(len(rows)):
            if isinstance(rows[i][0], list):
                new_row = []
                for r in rows[i][0]:
                    r = 1 / r
                    new_row.append(r)
                self.cursor.execute(f"update {self.vars[1]} set val = array{new_row} where row = {i + 1};")
                self.conn.commit()
                flag = 1
            else:
                val = rows[i][0]

        table_name = 'grad_' + str(self.id)
        if grad_output == 1:
            if flag == 1:
                self.cursor.execute(f"select * into {table_name} from {temp_table_name};")
            else:
                self.cursor.execute(f"create table {table_name}(row integer primary key, val double precision);")
                self.cursor.execute(f"insert into {table_name} values(1, {val});")
            self.conn.commit()
        else:
            self.cursor.execute(f"select val from {grad_output};")
            rows = self.cursor.fetchall()
            if isinstance(rows[0][0], list):
                if flag == 1:
                    self.cursor.execute(f"select db4ai_matmul('{temp_table_name}', '{grad_output}', '{table_name}'")
                else:
                    self.cursor.execute(f"select madlib.matrix_scalar_mult('{temp_table_name}', 'row=row,val=val',"
                                        f"'{val}', '{table_name}','row=row,val=val'")
            else:
                if flag == 1:
                    self.cursor.execute(f"select madlib.matrix_scalar_mult('{temp_table_name}', 'row=row,val=val',"
                                        f"'{rows[0][0]}', '{table_name}','row=row,val=val'")
                else:
                    self.cursor.execute(f"create table {table_name}(row integer primary key, val double precision);")
                    self.cursor.execute(f"insert into {table_name} values(1, {val * rows[0][0]});")
            self.conn.commit()
        return table_name


class POW(Node):
    def __init__(self, **kwargs):
        super().__init__(17, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # if self.physic_algorithm != 'madlib':
        #     self.executor.var_dict[self.vars[0]] = torch.pow(self.executor.var_dict[self.vars[1]],
        #                                                      self.executor.var_dict[self.vars[2]])
        # else:
        self.cursor.execute(f"select val from {self.vars[2]};")
        rows = self.cursor.fetchall()
        pow_exp = rows[0][0]
        self.cursor.execute(f"select db4ai_pow('{self.vars[1]}', {pow_exp}, '{self.vars[0]}');")
        self.conn.commit()

    def backward(self, grad_output=-1):

        self.cursor.execute(f"select val from {self.vars[2]};")
        rows = self.cursor.fetchall()
        pow_exp = rows[0][0]

        temp_name = 'grad_temp_' + self.id
        self.cursor.execute(f"select db4ai_pow('{self.vars[1]}', {self.vars[2] - 1}, {temp_name});")
        self.conn.commit()

        self.cursor.execute(f"select count(*) from pg_class where relname = {grad_output}")
        rows = self.cursor.fetchall()
        grad_name = 'grad_' + self.id
        if rows[0][0]:
            self.cursor.execute(f"select db4ai_mul('{temp_name}', {grad_output}, {grad_name});")
            self.conn.commit()
        else:
            self.cursor.execute(f"select * into {grad_name} from {temp_name};")
            self.conn.commit()

        self.cursor.execute(f"select val from {grad_name}")
        rows = self.cursor.fetchall()
        for i in range(len(rows)):
            if isinstance(rows[i][0], list):
                new_val = []
                for i in rows[i][0]:
                    i = i * pow_exp
                    new_val.append(i)
                self.cursor.execute(f"update {grad_name} set val = array{new_val} where id = {i + 1}")
                self.conn.commit()
            else:
                self.cursor.execute(f"update {grad_name} set val = {rows[i][0] * pow_exp} where id = {i + 1}")
                self.conn.commit()
        return grad_name


class SQRT(Node):
    def __init__(self, **kwargs):
        super().__init__(18, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        '''if self.physic_algorithm != 'madlib':
            self.executor.var_dict[self.vars[0]] = torch.sqrt(self.executor.var_dict[self.vars[1]])
        else:
            pass'''
        self.cursor.execute(f"select db4ai_sqrt('{self.vars[1]}', '{self.vars[0]}');")
        self.conn.commit()

    def backward(self, grad_output=1):
        temp_table_name = 'temp_grad_' + str(self.id)
        self.cursor.execute(f"drop table if exits {temp_table_name};")
        self.cursor.execute(f"insert * into {temp_table_name} from {self.vars[0]};")
        self.cursor.execute(f"select * from {self.vars[0]};")
        rows = self.cursor.fetchall()
        for i in range(len(rows)):
            new_row = []
            for r in rows[i][0]:
                r = 0.5 / r
                new_row.append(r)
            self.cursor.execute(f"update {self.vars[0]} set val = {new_row} where id = {i + 1};")
        if grad_output != 1:
            table_name = 'grad_' + str(self.id)
            self.cursor.execute(f"drop table if exits {table_name};")
            self.cursor.execute(f"select madlib.matrix_mult('{temp_table_name}', 'row=row, val=val',"
                                f"'{grad_output}', 'row=row, val=val', '{table_name}');")
            self.conn.commit()
            return table_name
        else:
            self.conn.commit()
            return temp_table_name


class MATMUL(Node):
    def __init__(self, **kwargs):
        super().__init__(19, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.matmul(self.executor.var_dict[self.vars[1]],
        #                                                     self.executor.var_dict[self.vars[2]])
        self.cursor.execute(f"select db4ai_matmul('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        self.conn.commit()

    def backward(self, grad_output=1):
        if grad_output == 1:
            grad_output_1 = 'grad_' + str(self.id) + '_temp1'
            grad_output_2 = 'grad_' + str(self.id) + '_temp2'
            self.cursor.execute(f"select val from {self.vars[1]}';")
            rows = self.cursor.fetchall()
            row_num_1 = len(rows)
            column_num_1 = len(rows[0][0])
            self.cursor.execute(f"select val from {self.vars[2]}';")
            rows = self.cursor.fetchall()
            row_num_2 = len(rows)
            column_num_2 = len(rows[0][0])
            self.cursor.execute(f"select db4ai_ones('{row_num_1}', '{column_num_1}', '{grad_output_1}');")
            self.cursor.execute(f"select db4ai_ones('{row_num_2}', '{column_num_2}', '{grad_output_2}');")
            self.conn.commit()
            '''grad_output_1 = torch.ones_like(self.executor.var_dict[self.vars[2]].T)
            grad_output_2 = torch.ones_like(self.executor.var_dict[self.vars[1]].T)'''
        else:
            grad_output_1 = grad_output
            grad_output_2 = grad_output
        trans_table_1 = self.vars[1] + '_T'
        trans_table_2 = self.vars[2] + '_T'
        table_1 = 'grad_' + str(self.id) + '_1'
        table_2 = 'grad_' + str(self.id) + '_2'
        self.cursor.execute(f"drop table if exists {trans_table_1}');")
        self.cursor.execute(f"drop table if exists {trans_table_2}');")
        self.cursor.execute(f"select madlib.matrix_trans('{self.vars[1]}', 'row=row, val=val','{trans_table_1}');")
        self.cursor.execute(f"select madlib.matrix_trans('{self.vars[2]}', 'row=row, val=val','{trans_table_2}');")
        self.cursor.execute(f"select db4ai_matmul('{grad_output_1}', '{trans_table_2}', '{table_1}');")
        self.cursor.execute(f"select db4ai_matmul('{trans_table_1}', '{grad_output_2}', '{table_2}');")
        self.conn.commit()
        return table_1, table_2


class DOT(Node):
    def __init__(self, **kwargs):
        super().__init__(20, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.dot(self.executor.var_dict[self.vars[1]],
        #                                                  self.executor.var_dict[self.vars[2]])
        self.cursor.execute(f"select db4ai_dot('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        self.conn.commit()


class INNER(Node):
    def __init__(self, **kwargs):
        super().__init__(21, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.inn(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]])
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
        # self.executor.var_dict[self.vars[0]] = torch.tensordot(self.executor.var_dict[self.vars[1]],
        #                                                        self.executor.var_dict[self.vars[2]])
        self.cursor.execute(f"select db4ai_tensordot('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        self.conn.commit()


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
        self.executor.var_dict[self.vars[0]] = torch.linalg.svd(self.executor.var_dict[self.vars[1]],
                                                                full_matrices=self.full_matrices,
                                                                compute_uv=self.compute_uv)


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
        self.executor.var_dict[self.vars[0]] = torch.det(self.executor.var_dict[self.vars[1]])


class RANK(Node):
    def __init__(self, **kwargs):
        super().__init__(31, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.rank(self.executor.var_dict[self.vars[1]])
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
        # self.executor.var_dict[self.vars[0]] = torch.trace(self.executor.var_dict[self.vars[1]])
        self.cursor.execute(f"select db4ai_trace('{self.vars[1]}', '{self.vars[0]}');")
        self.conn.commit()


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
        # self.executor.var_dict[self.vars[0]] = torch.reshape(self.executor.var_dict[self.vars[1]], self.new_shape)
        self.cursor.execute(
            f"select db4ai_reshape('{self.vars[1]}', {self.new_shape[0]}, {self.new_shape[1]}, '{self.vars[0]}');")
        self.conn.commit()


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
        self.cursor.execute("select val from grad_" + self.vars[0])
        rows = self.cursor.fetchall()
        print(rows)


class SHAPE(Node):
    def __init__(self, **kwargs):
        super().__init__(37, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.tensor(self.executor.var_dict[self.vars[1]].shape)
        self.cursor.execute(f"select db4ai_shape('{self.vars[1]}', '{self.vars[0]}');")
        self.conn.commit()


class EXP(Node):
    def __init__(self, **kwargs):
        super().__init__(38, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.exp(self.executor.var_dict[self.vars[1]])
        self.cursor.execute(f"select db4ai_exp('{self.vars[1]}', '{self.vars[0]}');")
        self.conn.commit()

    def backward(self, grad_output=1):
        temp_table_name = 'grad_' + str(self.id) + '_2'
        self.cursor.execute(f"select db4ai_exp('{self.vars[1]}', '{temp_table_name}');")
        table_name = 'grad_' + str(self.id)
        self.cursor.execute(f"select db4ai_mul('{grad_output}','{temp_table_name}', '{table_name}');")
        self.conn.commit()
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
        # s = fill_slice_var(self.slice_index, self.executor)
        # self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]].__getitem__(s)
        s = copy(self.slice_index)
        for idx in range(len(s)):
            if isinstance(s[idx], str):
                s[idx] = self.cursor.execute(f"select val from {s[idx]};")
            elif isinstance(s[idx], slice):
                start = s[idx].start
                stop = s[idx].stop
                if isinstance(s[idx].start, str):
                    start = self.cursor.execute(f"select val from {s[idx].start};")
                if isinstance(s[idx].stop, str):
                    stop = self.cursor.execute(f"select val from {s[idx].stop};")
                s[idx] = [start, stop]
        self.cursor.execute(
            f"select db4ai_slice({self.vars[1]}, {s[0][0]}, {s[0][1]}, {s[1][0]}, {s[1][1]}, {self.vars[0]});")
        self.conn.commit()


# 该类用来存储参数变量，如x，y
class Var(Node):
    def __init__(self, **kwargs):
        super().__init__(40, **kwargs)
        if 'vars' in kwargs.keys():
            self.set_vars(kwargs['vars'])


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
        # self.executor.var_dict[self.vars[0]] = torch.argmax(self.executor.var_dict[self.vars[1]], self.axis)
        self.cursor.execute(f"select db4ai_argmax('{self.vars[1]}', self.axis, '{self.vars[0]}');")
        self.conn.commit()

    def set_axis(self, axis):
        self.axis = axis


class Argmin(Node):
    def __init__(self, **kwargs):
        super().__init__(45, **kwargs)
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.argmin(self.executor.var_dict[self.vars[1]], self.axis)
        self.cursor.execute(f"select db4ai_argmin('{self.vars[1]}', self.axis, '{self.vars[0]}');")
        self.conn.commit()

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
        # if isinstance(self.data_shape, str):
        # 运行时使用变量的值填充变量名
        # for name in self.data_shape_var.keys():
        # self.data_shape_var[name] = int(self.executor.var_dict[name])
        # 转换
        # self.data_shape = eval(self.data_shape, self.data_shape_var)
        # self.executor.var_shape[self.vars[0]] = self.data_shape
        # tensor = torch.full(self.data_shape, self.num)
        # if self.with_grad:
        #     tensor.requires_grad = True
        # self.executor.var_dict[self.vars[0]] = tensor
        if isinstance(self.data_shape, str):
            # 运行时使用变量的值填充变量名
            for name in self.data_shape_var.keys():
                self.data_shape_var[name] = self.cursor.execute(f"select val from {name}")
            # 转换
            self.data_shape = eval(self.data_shape, self.data_shape_var)
        self.cursor.execute(
            f"select db4ai_full({self.data_shape[0]}, {self.data_shape[1]}, {self.num}, '{self.vars[0]}');")
        self.conn.commit()

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
        # if isinstance(self.data_shape, str):
        #     # 运行时使用变量的值填充变量名
        #     for name in self.data_shape_var.keys():
        #         self.data_shape_var[name] = int(self.executor.var_dict[name])
        #     # 转换
        #     self.data_shape = eval(self.data_shape, self.data_shape_var)
        # self.executor.var_shape[self.vars[0]] = self.data_shape
        # tensor = torch.ones(self.data_shape)
        # if self.with_grad:
        #     tensor.requires_grad = True
        # self.executor.var_dict[self.vars[0]] = tensor
        if isinstance(self.data_shape, str):
            # 运行时使用变量的值填充变量名
            for name in self.data_shape_var.keys():
                self.data_shape_var[name] = self.cursor.execute(f"select val from {name};")
            # 转换
            self.data_shape = eval(self.data_shape, self.data_shape_var)
        # self.vars[0] = "p" + self.vars[0]
        self.cursor.execute(f"select db4ai_ones({self.data_shape[0]}, {self.data_shape[1]}, '{self.vars[0]}');")
        self.conn.commit()

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
        # if isinstance(self.data_shape, str):
        #     # 运行时使用变量的值填充变量名
        #     for name in self.data_shape_var.keys():
        #         self.data_shape_var[name] = int(self.executor.var_dict[name])
        #     # 转换
        #     self.data_shape = eval(self.data_shape, self.data_shape_var)
        # self.executor.var_shape[self.vars[0]] = self.data_shape
        # tensor = torch.zeros(self.data_shape)
        # if self.with_grad:
        #     tensor.requires_grad = True
        # self.executor.var_dict[self.vars[0]] = tensor
        if isinstance(self.data_shape, str):
            # 运行时使用变量的值填充变量名
            for name in self.data_shape_var.keys():
                self.data_shape_var[name] = self.cursor.execute(f"select val from {name};")
            # 转换
            self.data_shape = eval(self.data_shape, self.data_shape_var)
        self.cursor.execute(f"select db4ai_zeros({self.data_shape[0]}, {self.data_shape[1]}, '{self.vars[0]}');")
        self.conn.commit()

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
        self.axis = 2

    def set_axis(self, axis):
        self.axis = axis

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.sum(self.executor.var_dict[self.vars[1]], self.axis)
        self.cursor.execute(f"select db4ai_sum('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")
        self.conn.commit()

    def backward(self, grad_output=1):
        return grad_output


class Relu(Node):
    def __init__(self, **kwargs):
        super().__init__(51, **kwargs)

    '''参数待补充'''

    def backward(self, grad_output):
        input_x, = self.saved_tensors
        if input_x < 0:
            grad_x = grad_output * 0
        else:
            grad_x = grad_output
        return grad_x


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
        # self.executor.var_dict[self.vars[0]] = torch.softmax(self.executor.var_dict[self.vars[1]], self.dim)
        self.cursor.execute(f"select db4ai_softmax('{self.vars[1]}', {self.dim}, '{self.vars[0]}');")
        self.conn.commit()


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
        # self.executor.var_dict[self.vars[0]] = torch.mean(self.executor.var_dict[self.vars[1]])
        self.cursor.execute(f"select db4ai_mean('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")
        self.conn.commit()

    def set_axis(self, axis):
        self.axis = axis

    def backward(self, grad_output=1):
        return 1 / self.executor.var_dict[self.vars[1]].numel() * grad_output


class MAX(Node):
    def __init__(self, **kwargs):
        super().__init__(58, **kwargs)
        self.axis = 2

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.max(self.executor.var_dict[self.vars[1]])
        self.cursor.execute(f"select db4ai_max('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")
        self.conn.commit()

    def set_axis(self, axis):
        self.axis = axis


class MIN(Node):
    def __init__(self, **kwargs):
        super().__init__(59, **kwargs)
        self.axis = 2

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.min(self.executor.var_dict[self.vars[1]])
        self.cursor.execute(f"select db4ai_min('{self.vars[1]}', {self.axis}, '{self.vars[0]}');")
        self.conn.commit()

    def set_axis(self, axis):
        self.axis = axis


class Abs(Node):
    def __init__(self, **kwargs):
        super().__init__(60, **kwargs)
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.abs(self.executor.var_dict[self.vars[1]])
        self.cursor.execute(f"select db4ai_abs('{self.vars[1]}', '{self.vars[0]}');")
        self.conn.commit()

    def set_axis(self, axis):
        self.axis = axis


class SplitDataset(Node):
    def __init__(self, **kwargs):
        # SplitDataset(data:variable, size:tensor)
        super().__init__(61, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        res = torch.utils.data.random_split(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]])
        self.executor.var_dict[self.vars[0]] = torch.stack([tensor for tensor in res])



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
        paras = self.executor.parameters[self.set_name]
        tmp = {}
        for para in paras:
            tmp[para] = self.executor.var_dict[para]
        dump_tensor(tmp, self.path)


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
        paras = self.executor.parameters[self.set_name]
        tensors = load_tensor(self.path)
        for para in paras:
            self.executor.var_dict[para] = tensors[para]


class AUC(Node):
    def __init__(self, **kwargs):
        super().__init__(65, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        test_y = self.executor.var_dict[self.vars[1]]
        pred = self.executor.var_dict[self.vars[2]]
        if len(test_y.shape) == 2 and len(pred.shape) == 1:
            pred = torch.unsqueeze(pred, len(pred.shape))
        if len(torch.unique(self.executor.var_dict[self.vars[1]])) > 2:
            self.executor.var_dict[self.vars[0]] = torch.tensor(
                sk_metrics.roc_auc_score(test_y, pred, multi_class='ovr'))
        else:
            self.executor.var_dict[self.vars[0]] = torch.tensor(sk_metrics.roc_auc_score(test_y, pred))


class MSE(Node):
    def __init__(self, **kwargs):
        super().__init__(66, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.tensor(
            sk_metrics.mean_squared_error(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]]))


class F1(Node):
    def __init__(self, **kwargs):
        super().__init__(66, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.tensor(
            sk_metrics.f1_score(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]],
                                average='macro'))


class REVERSE(Node):
    def __init__(self, **kwargs):
        super().__init__(67, **kwargs)
        # TODO: dims

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.flip(self.executor.var_dict[self.vars[1]], (0,))
        self.cursor.execute(f"select db4ai_reverse('{self.vars[1]}', 1, '{self.vars[0]}');")
        self.conn.commit()


class ARGSORT(Node):
    def __init__(self, **kwargs):
        super().__init__(68, **kwargs)
        self.dim = 2

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.argsort(self.executor.var_dict[self.vars[1]])
        self.cursor.execute(f"select db4ai_argsort('{self.vars[1]}', {self.dim}, '{self.vars[0]}');")
        self.conn.commit()

    def set_axis(self, dim):
        self.dim = dim


class SORT(Node):
    def __init__(self, **kwargs):
        super().__init__(69, **kwargs)
        self.dim = 2

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.sort(self.executor.var_dict[self.vars[1]])[0]
        self.cursor.execute(f"select db4ai_sort('{self.vars[1]}', {self.dim}, '{self.vars[0]}');")
        self.conn.commit()

    def set_dim(self, dim):
        self.dim = dim


class ACC(Node):
    def __init__(self, **kwargs):
        super().__init__(70, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.tensor(sk_metrics.accuracy_score(self.executor.var_dict[
        # self.vars[1]], self.executor.var_dict[self.vars[2]]))
        self.cursor.execute(f"select db4ai_acc('{self.vars[1]}', 'P', '{self.vars[0]}');")
        self.conn.commit()


class RECALL(Node):
    def __init__(self, **kwargs):
        super().__init__(71, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.tensor(sk_metrics.recall_score(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]], average='macro'))
        self.cursor.execute(f"select db4ai_recall('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        self.conn.commit()


class PRECISION(Node):
    def __init__(self, **kwargs):
        super().__init__(72, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.tensor(sk_metrics.precision_score(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]], average='macro'))
        self.cursor.execute(f"select db4ai_precision('{self.vars[1]}', '{self.vars[2]}', '{self.vars[0]}');")
        self.conn.commit()


"""
Backward计算叶节点梯度，存放于Node_grad表中
"""


class Backward(Node):
    def __init__(self, **kwargs):
        super().__init__(73, **kwargs)

    @preprocessing
    def run(self, **kwargs):

        # self.executor.var_dict[self.vars[0]].backward()
        """
        初始化
        """
        nodes = []
        for pre_node in self.pre_nodes():
            nodes.append(pre_node)
        """
        遍历所有点 
        """
        while nodes:
            node = nodes.pop()
            # 计算算子梯度
            if node.__class__.__name__ in all_operator:
                table_name = 'grad_' + str(node.id)
                self.cursor.execute(f"select count(*) from pg_class where relname = '{table_name}'")
                node_flag = self.cursor.fetchall()[0][0] == 1
                if node_flag:
                    tup = node.backward('grad_' + str(node.id))
                else:
                    tup = node.backward()
                if not isinstance(tup, tuple):
                    tup = (tup,)
                # id由小到大排序父节点
                fathers = []
                for pre_node in node.pre_nodes():
                    index = 0
                    for father in fathers:
                        if pre_node.id > father.id:
                            index += 1
                    fathers.insert(index, pre_node)
                index = 0
                for father in fathers:
                    if not isinstance(father, Root):
                        if isinstance(father, Var) or father.__class__.__name__ in all_operator:
                            if father.with_grad is True:
                                fa_table_name = "grad_" + str(father.id)
                                self.cursor.execute(f"select count(*) from pg_class where relname = '{fa_table_name}'")
                                node_flag = self.cursor.fetchall()[0][0] == 0
                                if node_flag:
                                    if tup[index] == 1:
                                        self.cursor.execute(f"create table {fa_table_name}(row integer primary key,"
                                                            f"val double precision)")
                                        self.cursor.execute(f"insert into {fa_table_name} values (1,1.0)")
                                    else:
                                        self.cursor.execute(f"select * into {fa_table_name} from {tup[index]}")
                                '''else:
                                        father.grad *= tup[index]'''
                                self.conn.commit()
                        index += 1
                        nodes.append(father)
                self.cursor.execute(f"drop table if exists {table_name}")
                self.cursor.execute(f"drop table if exists {table_name + '_1'}")
                self.cursor.execute(f"drop table if exists {table_name + '_2'}")
                self.cursor.execute(f"drop table if exists {table_name + '_temp1'}")
                self.cursor.execute(f"drop table if exists {table_name + '_temp2'}")
                self.conn.commit()
                '''node.grad = None
                table_name = "grad_" + node.id
                '''
            # 继承梯度
            else:
                for father in node.pre_nodes():
                    if not isinstance(node, CreateTensor):
                        table_name = "grad_" + str(node.id)
                    else:
                        table_name = "grad_" + node.vars[0]
                    if not isinstance(father, Root):
                        self.cursor.execute(f"select count(*) from pg_class where relname = '{table_name}'")
                        node_flag = self.cursor.fetchall()[0][0] == 1
                        if node_flag:
                            if not isinstance(father, CreateTensor):
                                fa_table_name = "grad_" + str(father.id)
                            else:
                                fa_table_name = "grad_" + father.vars[0]
                            self.cursor.execute(f"select count(*) from pg_class where relname = '{fa_table_name}'")
                            father_flag = self.cursor.fetchall()[0][0] == 1
                            if isinstance(father, Assignment) and father_flag:
                                self.cursor.execute(f"select val from {table_name}")
                                rows = self.cursor.fetchall()
                                self.cursor.execute(f"select val from {fa_table_name}")
                                fa_rows = self.cursor.fetchall()
                                for row in fa_rows:
                                    new_row = numpy.array(row[0]) + numpy.array(rows[list(fa_rows).index(row)])
                                    self.cursor.execute(f"update {fa_table_name} set val = array{list(new_row)} where "
                                                        f"row = {list(fa_rows).index(row) + 1}")
                            else:
                                self.cursor.execute(f"drop table if exists {fa_table_name}")
                                self.cursor.execute(f"select * into {fa_table_name} from {table_name}")
                        nodes.append(father)
                if not isinstance(node, CreateTensor):
                    self.cursor.execute(f"drop table if exists {table_name}")
                    self.conn.commit()


class WLS(Node):
    def __init__(self, **kwargs):
        super().__init__(74, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.tensor(sklearn.linear_model.LinearRegression(
            self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]],
            sample_weight=self.executor.var_dict[self.vars[2]]))


class REPEAT(Node):
    def __init__(self, **kwargs):
        super().__init__(75, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]].repeat(self.executor.var_dict[self.vars[2]],self.executor.var_dict[self.vars[3]],self.executor.var_dict[self.vars[4]])
        self.cursor.execute(
            f"select db4ai_repeat('{self.vars[1]}', '{self.vars[4]}', '{self.vars[3]}', {self.vars[0]}');")
        self.conn.commit()

    def backward(self, grad_output=1):
        temp = self.executor.var_dict[self.vars[2]] * self.executor.var_dict[self.vars[3]] * self.executor.var_dict[
            self.vars[4]] * grad_output
        return torch.Tensor(temp, temp, temp)


class UNSQUEEZE(Node):
    def __init__(self, **kwargs):
        super().__init__(76, **kwargs)
        self.dim = None

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.unsqueeze(self.executor.var_dict[self.vars[1]], self.dim)

    def set_dim(self, dim):
        self.dim = int(dim)


class CleanGrad(Node):
    def __init__(self, **kwargs):
        super().__init__(77, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        '''for name in self.vars:
            if self.executor.var_dict[name].grad is not None:
                self.executor.var_dict[name].grad.data.zero_()'''
        self.cursor.execute(f"drop table if exists {'grad_' + self.vars[0]}")
        self.conn.commit()


class Negative(Node):
    def __init__(self, **kwargs):
        super().__init__(78, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.cursor.execute(f"select * into {self.vars[1]} from {self.vars[0]}")
        self.cursor.execute(f"select * from {self.vars[1]}")
        rows = self.cursor.fetchall()
        update = []
        for i in range(len(rows)):
            update[i] = -1 * rows[i][1]
            self.cursor.execute(f"update {self.vars[1]} set val = {update[i]} where rows = {rows[i][0]}")
        self.conn.commit()

    def backward(self, grad_output):
        return -1 * grad_output


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
