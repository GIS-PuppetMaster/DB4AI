from Executor import *
from threading import Thread
import numpy as np
import torch
from functools import wraps
from copy import copy


class Node:
    # 计算图中节点类的父类
    def __init__(self, type_id, with_grad=False, physic_algorithm='relational', **kwargs):
        self.physic_algorithm = physic_algorithm
        self.id = kwargs['id']
        self.type_id = type_id
        self.with_grad = with_grad
        self.out_edges = []
        self.in_edges = []
        self.input_data_edges = []
        self.branches = kwargs['branches']
        self.vars = []
        self.executor = None
        self.batch_counter = 0
        self._default_batch_size = 0
        self.batch_size = 0
        self.use_batch = True
        self.fathers = []
        self.sons = []

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

    def run(self):
        pass

    def next_nodes(self):
        return self.sons

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
        self.vars.append(input)

    def get_vars(self):
        return self.vars

    def __repr__(self):
        return f'id:{self.id}, branches:{self.branches}, vars:{self.vars}'


# 通过继承实现的其它节点类
class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)


# 创建张量所用节点
class CreateTensor(Node):
    def __init__(self, data_shape, var, **kwargs):
        super().__init__(1, **kwargs)
        if isinstance(data_shape, tuple):
            self.data_shape = data_shape
        else:
            self.data_shape = eval(data_shape)
        self.var = var

    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.empty(size=self.data_shape, requires_grad=self.with_grad)
        # self.executor.var_dict[self.vars[0]] = None
        pass

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape

    def get_val(self):
        return self.var


class Val(Node):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)
        self.value = 0

    def set_val(self, value):
        self.value = value

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.tensor(self.value)

    def get_val(self):
        return self.value

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = (1,)


class Sql(Node):
    def __init__(self, t_info, **kwargs):
        super().__init__(3, **kwargs)
        self.t_search_sentences = t_info
        self.shape = None
        # self.batch_size = None  # TODO: 自动选择batch_size

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = None  # TODO:get data

    def infer_data(self):
        for edge in self.out_edges:
            # TODO: 使用SQL查询该表的维度
            self.shape = None
            edge.data_shape = self.shape


class Random(Node):
    def __init__(self, boundary, data_shape, distribution=None, **kwargs):
        super().__init__(4, **kwargs)
        self.boundary = boundary
        if isinstance(data_shape, tuple):
            self.data_shape = data_shape
        else:
            self.data_shape = eval(data_shape)
        if distribution == '' or distribution is None or (isinstance(distribution, list) and len(distribution) == 0):
            self.distribution = 'normal'
        else:
            self.distribution = distribution

    def run(self, **kwargs):
        if self.distribution == 'normal':
            # boundary[0]=lower_boundary, boundary[1]=upper_boundary
            tensor = torch.randn(self.data_shape) * (self.boundary[1] - self.boundary[0]) + self.boundary[0]
        elif self.distribution == 'gauss':
            # boundary[0]=mu, boundary[1]=sigma
            tensor = torch.randn() * self.boundary[1] + self.boundary[0]
        else:
            raise Exception(f'Not supported distribution:{self.distribution}')
        self.executor.var_dict[self.vars[0]] = tensor

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_type = 'ndarray'
            edge.data_shape = self.data_shape


# 逻辑控制所用节点
class Loop(Node):
    def __init__(self, condition, loop_id, **kwargs):
        super().__init__(6, **kwargs)
        if condition:
            self.dead_cycle = condition
            self.times = 0
        else:
            self.dead_cycle = False
            self.times = condition
        self.loop_id = loop_id
        self.finished_times = 0

    def next_nodes(self):
        end_nodes = [edge.end for edge in self.out_edges]
        last_nodes = [edge.start for edge in self.in_edges]
        loop_end_node = None
        for node in last_nodes:
            if isinstance(node, LoopEnd) and node.loop_id == self.loop_id:
                loop_end_node = node
                break
        assert loop_end_node is not None, f'Did not find corresponding loop end node for loop node{self.loop_id}'
        if loop_end_node in end_nodes:
            end_nodes.remove(loop_end_node)
        # 循环结束
        if self.finished_times >= self.times:
            # 找到对应的Loop_End
            self.executor.finished_loop_id.add(self.loop_id)
            return [loop_end_node]
        else:
            return end_nodes


class LoopEnd(Node):
    def __init__(self, loop_id, **kwargs):
        super().__init__(7, **kwargs)
        self.loop_id = loop_id

    def next_nodes(self):
        end_nodes = [edge.end for edge in self.out_edges]
        loop_node = None
        for node in end_nodes:
            if isinstance(node, Loop) and node.loop_id == self.loop_id:
                loop_node = node
                break
        assert loop_node is not None, f'Did not find corresponding loop node for end loop node{self.loop_id}'
        end_nodes.remove(loop_node)
        # 退出循环
        if self.loop_id in self.executor.finished_loop_id:
            self.executor.finished_loop_id.remove(self.loop_id)
            return end_nodes
        # 继续下一次循环
        else:
            return [loop_node]


class Break(Node):
    def __init__(self, loop_id, **kwargs):
        super().__init__(8, **kwargs)
        self.loop_id = loop_id

    def next_nodes(self):
        self.executor.finished_loop_id.remove(self.loop_id)
        end_nodes = [edge.end for edge in self.out_edges]
        loop_end_node = None
        for node in end_nodes:
            if isinstance(node, LoopEnd) and node.loop_id == self.loop_id:
                loop_end_node = node
                break
        assert loop_end_node is not None, f'Did not find corresponding loop end node for break node{self.loop_id}'
        return loop_end_node


class If(Node):
    def __init__(self, **kwargs):
        super().__init__(9, **kwargs)

    def next_nodes(self):
        for edge in self.out_edges:
            para = {}
            for var_name, var_node in edge.need_var:
                para[var_name] = var_node
            res = eval(edge.condition, para)
            if edge.reverse:
                res = not res
            if res:
                return [edge.end]


class IfBranch(Node):
    def __init__(self, **kwargs):
        super().__init__(10, **kwargs)

    def next_nodes(self):
        for edge in self.out_edges:
            para = {}
            for var_name, var_node in edge.need_var:
                para[var_name] = var_node
            res = eval(edge.condition, para)
            if edge.reverse:
                res = not res
            if res:
                return [edge.end]


class IfEnd(Node):
    def __init__(self, **kwargs):
        super().__init__(11, **kwargs)


class Assignment(Node):
    def __init__(self, var_li, **kwargs):
        super().__init__(12, **kwargs)
        self.vars = var_li

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]]


class Add(Node):
    def __init__(self, **kwargs):
        super().__init__(12, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] + self.executor.var_dict[
            self.vars[2]]


class Sub(Node):
    def __init__(self, **kwargs):
        super().__init__(13, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] - self.executor.var_dict[
            self.vars[2]]


class Mul(Node):
    def __init__(self, **kwargs):
        super().__init__(14, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] * self.executor.var_dict[
            self.vars[2]]


class Div(Node):
    def __init__(self, **kwargs):
        super().__init__(15, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] / self.executor.var_dict[
            self.vars[2]]


class LOG(Node):
    def __init__(self, **kwargs):
        super().__init__(16, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.log(self.executor.var_dict[self.vars[1]])


class POW(Node):
    def __init__(self, **kwargs):
        super().__init__(17, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.pow(self.executor.var_dict[self.vars[1]],
                                                         self.executor.var_dict[self.vars[2]])


class SQRT(Node):
    def __init__(self, **kwargs):
        super().__init__(18, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.sqrt(self.executor.var_dict[self.vars[1]])


class MATMUL(Node):
    def __init__(self, **kwargs):
        super().__init__(19, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.matmul(self.executor.var_dict[self.vars[1]],
                                                            self.executor.var_dict[self.vars[2]])


class DOT(Node):
    def __init__(self, **kwargs):
        super().__init__(20, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.dot(self.executor.var_dict[self.vars[1]],
                                                         self.executor.var_dict[self.vars[2]])


class INNER(Node):
    def __init__(self, **kwargs):
        super().__init__(21, **kwargs)

    def run(self, **kwargs):
        # self.executor.var_dict[self.vars[0]] = torch.inn(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]])
        raise Exception('暂不支持inner')


class OUTER(Node):
    def __init__(self, **kwargs):
        super().__init__(22, **kwargs)

    def run(self, **kwargs):
        raise Exception('暂不支持outer')


class TENSORDOT(Node):
    def __init__(self, **kwargs):
        super().__init__(23, **kwargs)

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.tensordot(self.executor.var_dict[self.vars[1]],
                                                               self.executor.var_dict[self.vars[2]])


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

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.det(self.executor.var_dict[self.vars[1]])


class RANK(Node):
    def __init__(self, **kwargs):
        super().__init__(31, **kwargs)

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

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.trace(self.executor.var_dict[self.vars[1]])


class RESHAPE(Node):
    def __init__(self, **kwargs):
        super().__init__(33, **kwargs)
        self.new_shape = None
        self.order = 'C'

    def set_param(self, newshape, order):
        self.new_shape = newshape
        self.order = order

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.reshape(self.executor.var_dict[self.vars[1]], self.new_shape)


class TRANSPOSE(Node):
    def __init__(self, **kwargs):
        super().__init__(34, **kwargs)


class STACK(Node):
    def __init__(self, **kwargs):
        super().__init__(35, **kwargs)
        self.axis = 0

    def set_axis(self, axis):
        self.axis = axis


# 该类实例含义为当前位置值未知，占空，之后被其他类实例取代
class Blank(Node):
    def __init__(self, **kwargs):
        super().__init__(36, **kwargs)


# 该类为列表切片、索引，self.name为列表名，self.slice_info为切片信息
class Slice(Node):
    def __init__(self, **kwargs):
        super().__init__(37, **kwargs)
        self.name = ''
        self.slice_info = None
        self.slice_index = None

    def set_name(self, name):
        self.name = name

    def set_slice(self, slice_info):
        self.slice_info = slice_info
        total_slice = []
        for idx in self.slice_info:
            if ':' in idx:
                total_slice.append(slice(*list(map(lambda x: None if x == '' else int(x), idx.split(':')))))
            else:
                total_slice.append(int(idx))
        self.slice_index = total_slice

    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]].__getitem__(self.slice_index)


# 该类用来存储参数变量，如x，y
class Var(Node):
    def __init__(self, **kwargs):
        super().__init__(38, **kwargs)
        self.var = 0

    def set_val(self, var):
        self.var = var

    def get_val(self):
        return self.var


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
    if nodeType == 'CreateTensor':
        data_shape = otherField['data_shape']
        var = otherField['var']
        node = globals()[nodeType](data_shape, var, id=nodeId, branches=branches, with_grad=with_grad)
    elif nodeType == 'Sql':
        t_info = otherField['t_info']
        node = globals()[nodeType](t_info, id=nodeId, branches=branches, with_grad=with_grad)
    elif nodeType == 'Random':
        boundary = otherField['boundary']
        data_shape = otherField['data_shape']
        type = otherField['type']
        node = globals()[nodeType](boundary, data_shape, type, id=nodeId, branches=branches, with_grad=with_grad)
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
        node = globals()[nodeType](id=nodeId, branches=branches, with_grad=with_grad)
    return node
