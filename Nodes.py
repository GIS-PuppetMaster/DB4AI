import re
import torch
from functools import wraps
from copy import copy, deepcopy
import sklearn
import pickle as pk


def preprocessing(fun):
    @wraps(fun)
    def decorated(node, **kwargs):
        # todo 自动类型转换
        for input in node.vars[1:]:
            if node.physic_algorithm=='madlib' and not isinstance(node.executor.var_dict[input], str):
                # sql->torch
                pass
            elif node.physic_algorithm!='madlib' and isinstance(node.executor.var_dict[input],str):
                # torch->sql
                pass
        if not node.with_grad and not isinstance(node, GRADIENT):
            with torch.no_grad():
                return fun(node, **kwargs)
        else:
            return fun(node, **kwargs)

    return decorated


def dump_tensor(tensors: dict, path: str):
    with open(path, 'wb') as f:
        pk.dump(tensors, f)


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
            total_slice.append(slice(*list(map(lambda x: None if x == '' else (str(x) if re.fullmatch(re.compile(r'[a-zA-Z]+.*', re.S), x)
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
        if isinstance(input, list):
            self.vars = input
        else:
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
        # if isinstance(data_shape, tuple):
        #     self.data_shape = data_shape
        # elif isinstance(data_shape, str):
        #     self.data_shape = eval(data_shape)
        # elif data_shape is None:
        #     self.data_shape = None
        # TODO: infer data_shape
        self.set_vars(var)



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
        if self.vars[0] not in self.executor.var_dict:
            self.executor.var_dict[self.vars[0]] = torch.tensor(self.value)

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

    @preprocessing
    def run(self, **kwargs):
        if self.physic_algorithm != 'madlib':
            # 如果data shape中包含var
            if isinstance(self.data_shape, str):
                # 运行时使用变量的值填充变量名
                for name in self.data_shape_var.keys():
                    self.data_shape_var[name] = int(self.executor.var_dict[name])
                # 转换
                self.data_shape = eval(self.data_shape, self.data_shape_var)
            # 如果boundary中包含var
            if isinstance(self.boundary, str):
                # 运行时使用变量的值填充变量名
                for name in self.boundary_var.keys():
                    self.boundary_var[name] = int(self.executor.var_dict[name])
                # 转换
                self.boundary = eval(self.boundary, self.boundary_var)
            if self.distribution == 'normal':
                # boundary[0]=lower_boundary, boundary[1]=upper_boundary
                tensor = torch.randn(self.data_shape) * (self.boundary[1] - self.boundary[0]) + self.boundary[0]
            elif self.distribution == 'gauss':
                # boundary[0]=mu, boundary[1]=sigma
                tensor = torch.randn() * self.boundary[1] + self.boundary[0]
            elif self.distribution == 'int':
                tensor = torch.randint(low=self.boundary[0], high=self.boundary[1], size=self.data_shape)
            else:
                raise Exception(f'Not supported distribution:{self.distribution}')
            if self.with_grad:
                tensor.requires_grad = True
            self.executor.var_dict[self.vars[0]] = tensor
        else:
            # TODO:
            pass

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
            return [self.loop_pair]
        else:
            return end_nodes


class LoopEnd(Node):
    def __init__(self, loop_id, **kwargs):
        super().__init__(6, **kwargs)
        self.loop_id = loop_id
        self.loop_pair = None

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


class Assignment(Node):
    def __init__(self, var_li, **kwargs):
        super().__init__(11, **kwargs)
        self.vars = var_li
        self._slice = None
        if 'slice' in kwargs.keys():
            self.slice = kwargs['slice']

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
        right=self.executor.var_dict[self.vars[1]]
        left =self.executor.var_dict[self.vars[0]]
        if isinstance(right,torch.Tensor) :
            if left is None or isinstance(left,torch.Tensor):
                if self.slice is None:
                    self.executor.var_dict[self.vars[0]] = right
                else:
                    s = copy(self.slice)
                    for idx in range(len(s)):
                        if isinstance(s[idx], str):
                            s[idx] = int(self.executor.var_dict[s[idx]])
                    # if self.vars[0] not in self.executor.var_dict:
                    #     self.executor.var_dict[self.vars[0]] = torch.empty(self.executor.var_shape[self.vars[0]])
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
            if left is None or isinstance(left,str):
                # TODO: 赋值给madlib matrix
                self.executor.var_dict[self.vars[0]]=self.vars[0]
            else:
                # TODO: madlib_to_tensor，赋值
                pass

        if self.with_grad and not self.executor.var_dict[self.vars[0]].requires_grad:
            self.executor.var_dict[self.vars[0]].requires_grad = True




class Add(Node):
    def __init__(self, **kwargs):
        super().__init__(12, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        if self.physic_algorithm!='madlib':
            self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] + self.executor.var_dict[
                self.vars[2]]
        else:
            #TODO
            pass


class Sub(Node):
    def __init__(self, **kwargs):
        super().__init__(13, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        if self.physic_algorithm != 'madlib':
            self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] - self.executor.var_dict[
                self.vars[2]]
        else:
            #TODO
            pass

class Mul(Node):
    def __init__(self, **kwargs):
        super().__init__(14, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        if self.physic_algorithm != 'madlib':
            self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] * self.executor.var_dict[
                self.vars[2]]
        else:
            # TODO
            pass

class Div(Node):
    def __init__(self, **kwargs):
        super().__init__(15, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        if self.physic_algorithm != 'madlib':
            self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]] / self.executor.var_dict[
                self.vars[2]]
        else:
            # TODO
            pass


class LOG(Node):
    def __init__(self, **kwargs):
        super().__init__(16, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        if self.physic_algorithm != 'madlib':
            self.executor.var_dict[self.vars[0]] = torch.log(self.executor.var_dict[self.vars[1]])
        else:
            # TODO
            pass


class POW(Node):
    def __init__(self, **kwargs):
        super().__init__(17, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        if self.physic_algorithm != 'madlib':
            self.executor.var_dict[self.vars[0]] = torch.pow(self.executor.var_dict[self.vars[1]],
                                                             self.executor.var_dict[self.vars[2]])
        else:
            # TODO
            pass

class SQRT(Node):
    def __init__(self, **kwargs):
        super().__init__(18, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        if self.physic_algorithm != 'madlib':
            self.executor.var_dict[self.vars[0]] = torch.sqrt(self.executor.var_dict[self.vars[1]])
        else:
            # TODO
            pass


class MATMUL(Node):
    def __init__(self, **kwargs):
        super().__init__(19, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.matmul(self.executor.var_dict[self.vars[1]],
                                                            self.executor.var_dict[self.vars[2]])


class DOT(Node):
    def __init__(self, **kwargs):
        super().__init__(20, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.dot(self.executor.var_dict[self.vars[1]],
                                                         self.executor.var_dict[self.vars[2]])


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
        self.executor.var_dict[self.vars[0]] = torch.trace(self.executor.var_dict[self.vars[1]])


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


# 用来计算梯度
class GRADIENT(Node):
    def __init__(self, **kwargs):
        super().__init__(36, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        if len(self.vars) == 2:
            self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]].grad
        else:
            # TODO: add list data type
            self.executor.var_dict[self.vars[1]].backward()
            self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[2]].grad


class SHAPE(Node):
    def __init__(self, **kwargs):
        super().__init__(37, **kwargs)

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]].shape


class EXP(Node):
    def __init__(self, **kwargs):
        super().__init__(38, **kwargs)


# 该类为列表切片、索引，self.name为列表名，self.slice_info为切片信息
class Slice(Node):
    def __init__(self, **kwargs):
        super().__init__(39, **kwargs)
        self.slice_info = None
        self.slice_index = None

    def set_slice(self, slice_info):
        total_slice = parse_slice(slice_info)
        self.slice_index = total_slice

    @preprocessing
    def run(self, **kwargs):
        s = copy(self.slice_index)
        for idx in range(len(s)):
            if isinstance(s[idx], str):
                s[idx] = int(self.executor.var_dict[s[idx]])
        self.executor.var_dict[self.vars[0]] = self.executor.var_dict[self.vars[1]].__getitem__(s)


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
        self.executor.var_dict[self.vars[0]] = torch.argmax(self.executor.var_dict[self.vars[1]], self.axis)

    def set_axis(self, axis):
        self.axis = axis


class Argmin(Node):
    def __init__(self, **kwargs):
        super().__init__(45, **kwargs)
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.argmin(self.executor.var_dict[self.vars[1]], self.axis)

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
        if isinstance(data_shape, tuple):
            self.data_shape = data_shape
        elif isinstance(data_shape, str):
            self.data_shape = eval(data_shape)
        elif data_shape is None:
            self.data_shape = None
        # TODO: infer data_shape
        self.set_vars(var)
        self.num = num

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_shape[self.vars[0]] = self.data_shape
        tensor = torch.full(self.data_shape, self.num)
        if self.with_grad:
            tensor.requires_grad = True
        self.executor.var_dict[self.vars[0]] = tensor

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape


class Ones(Node):
    def __init__(self, data_shape, var, **kwargs):
        super().__init__(48, **kwargs)
        if isinstance(data_shape, tuple):
            self.data_shape = data_shape
        elif isinstance(data_shape, str):
            self.data_shape = eval(data_shape)
        elif data_shape is None:
            self.data_shape = None
        # TODO: infer data_shape
        self.set_vars(var)

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_shape[self.vars[0]] = self.data_shape
        tensor = torch.ones(self.data_shape)
        if self.with_grad:
            tensor.requires_grad = True
        self.executor.var_dict[self.vars[0]] = tensor

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape


class Zeros(Node):
    def __init__(self, data_shape, var, **kwargs):
        super().__init__(49, **kwargs)
        if isinstance(data_shape, tuple):
            self.data_shape = data_shape
        elif isinstance(data_shape, str):
            self.data_shape = eval(data_shape)
        elif data_shape is None:
            self.data_shape = None
        # TODO: infer data_shape
        self.set_vars(var)

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_shape[self.vars[0]] = self.data_shape
        tensor = torch.zeros(self.data_shape)
        if self.with_grad:
            tensor.requires_grad = True
        self.executor.var_dict[self.vars[0]] = tensor

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape


class SUM(Node):
    def __init__(self, **kwargs):
        super().__init__(50, **kwargs)
        self.axis = None

    def set_axis(self, axis):
        self.axis = axis

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.SUM(self.executor.var_dict[self.vars[1]], self.axis)


class Relu(Node):
    def __init__(self, **kwargs):
        super().__init__(51, **kwargs)


class Tanh(Node):
    def __init__(self, **kwargs):
        super().__init__(52, **kwargs)


class Softmax(Node):
    def __init__(self, **kwargs):
        super().__init__(53, **kwargs)


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
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.mean(self.executor.var_dict[self.vars[1]])

    def set_axis(self, axis):
        self.axis = axis


class MAX(Node):
    def __init__(self, **kwargs):
        super().__init__(58, **kwargs)
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.max(self.executor.var_dict[self.vars[1]])

    def set_axis(self, axis):
        self.axis = axis


class MIN(Node):
    def __init__(self, **kwargs):
        super().__init__(59, **kwargs)
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.min(self.executor.var_dict[self.vars[1]])

    def set_axis(self, axis):
        self.axis = axis


class Abs(Node):
    def __init__(self, **kwargs):
        super().__init__(60, **kwargs)
        self.axis = 0

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.abs(self.executor.var_dict[self.vars[1]])

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
        # TODO: 解析

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.tensor(sklearn.metrics.roc_auc_score(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]]))


class MSE(Node):
    def __init__(self, **kwargs):
        super().__init__(66, **kwargs)
        # TODO: 解析


    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.tensor(sklearn.metrics.mean_squared_error(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]]))


class F1(Node):
    def __init__(self, **kwargs):
        super().__init__(66, **kwargs)
        # TODO: 解析

    @preprocessing
    def run(self, **kwargs):
        self.executor.var_dict[self.vars[0]] = torch.tensor(sklearn.metrics.f1_score(self.executor.var_dict[self.vars[1]], self.executor.var_dict[self.vars[2]]))


class REVERSE(Node):
    def __init__(self, **kwargs):
        super().__init__(67, **kwargs)


class ARGSORT(Node):
    def __init__(self, **kwargs):
        super().__init__(68, **kwargs)


class SORT(Node):
    def __init__(self, **kwargs):
        super().__init__(69, **kwargs)


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
