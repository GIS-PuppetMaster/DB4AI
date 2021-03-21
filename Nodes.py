from Executor import *
from threading import Thread
import numpy as np

from Executor import BatchedTensor
from copy import copy


class Node(Thread):
    # 计算图中节点类的父类
    def __init__(self, id, physic_algorithm='relational', **kwargs):
        self.id = id
        super().__init__()
        self.physic_algorithm = physic_algorithm
        self.type_id = None
        self.with_grad = kwargs['with_grad']
        self.out_edges = []
        self.in_edges = []
        self.input_data_edges = []
        self.vars = []
        self.executor = None
        self.batch_counter = 0
        self._default_batch_size = 0
        self.batch_size = 0
        self.use_batch = True
        self.fathers = [edge.start for edge in self.in_edges]
        self.sons = [edge.end for edge in self.out_edges]
        self.branches = kwargs['branches']

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

    @batch_stream
    def run(self, input_buffer):
        # 默认转发经过的数据
        return input_buffer

    def next_nodes(self):
        return self.sons

    def infer_data(self):
        pass

    def GetType(self):
        return self.type_id

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        else:
            return False

    def __hash__(self):
        return hash(self.id)


# 通过继承实现的其它节点类
class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# 创建张量所用节点
class CreateTensor(Node):
    def __init__(self, data_shape, **kwargs):
        super().__init__(**kwargs)
        if data_shape:
            self.data_shape = eval(data_shape)
        else:
            self.data_shape = None
        self.use_batch = False

    def run(self, **kwargs):
        self.executor.pipeline[self.vars[0]].put(BatchedTensor(self.executor.var_dict[self.vars[0]], self.next_nodes(), batch_size=self.batch_size, start_index=0))

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape


class Val(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = 0
        self.use_batch = False

    def set_val(self, value):
        self.value = value

    def run(self, **kwargs):
        self.executor.graph.var_dict[self.vars[0]][...] = self.value
        data = self.executor.var_dict[self.vars[0]]
        self.executor.pipeline[self.vars[0]].put(BatchedTensor(data, self.next_nodes(), batch_size=self.batch_size, start_index=0))

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = (1,)


class Sql(Node):
    def __init__(self, t_info, **kwargs):
        super().__init__(**kwargs)
        self.t_search_sentences = t_info
        self.shape = None
        self.batch_size = None  # TODO: 自动选择batch_size

    def run(self, **kwargs):
        # 根据batch size划分
        for idx in range(0, np.prod(self.shape), self.batch_size):
            self.executor.var_dict[self.vars[0]][idx:idx + self.batch_size] = None  # TODO:get data
            for var_name in self.vars[1:]:
                self.executor.pipeline[var_name].put(BatchedTensor(self.executor.var_dict[self.vars[0]], self.next_nodes(), batch_size=self.batch_size, start_index=idx))

    def infer_data(self):
        for edge in self.out_edges:
            # TODO: 使用SQL查询该表的维度
            self.shape = None
            edge.data_shape = self.shape


class Random(Node):
    def __init__(self, boundary, data_shape, distribution, **kwargs):
        super().__init__(**kwargs)
        self.boundary = boundary
        self.data_shape = eval(data_shape)
        if distribution == '':
            self.distribution = 'normal'
        else:
            self.distribution = distribution
        self.use_batch = False

    def run(self, **kwargs):
        if self.distribution == 'normal':
            # boundary[0]=lower_boundary, boundary[1]=upper_boundary
            tensor = np.random.random(self.data_shape) * (self.boundary[1] - self.boundary[0]) + self.boundary[0]
        elif self.distribution == 'gauss':
            # boundary[0]=mu, boundary[1]=sigma
            tensor = np.random.randn() * self.boundary[1] + self.boundary[0]
        else:
            raise Exception(f'Not supported distribution:{self.distribution}')
        self.executor.graph.var_dict[self.vars[0]] = tensor
        data = self.executor.var_dict[self.vars[0]]
        self.executor.pipeline[self.vars[0]].put(BatchedTensor(data, self.next_nodes(), batch_size=self.batch_size, start_index=0))

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape


# 逻辑控制所用节点
class Loop(Node):
    def __init__(self, condition, loop_id, **kwargs):
        super().__init__(**kwargs)
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
        super().__init__(**kwargs)
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
        super().__init__(**kwargs)
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
        super().__init__(**kwargs)

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
        super().__init__(**kwargs)

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
        super().__init__(**kwargs)


class Assignment(Node):
    def __init__(self, var_li, **kwargs):
        super().__init__(**kwargs)
        self.var_li = var_li

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return input_buffer[0]()


class Add(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return input_buffer[0]() + input_buffer[1]()


class Sub(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return input_buffer[0]() - input_buffer[1]()


class Mul(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return input_buffer[0]() * input_buffer[1]()


class Div(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return input_buffer[0]() / input_buffer[1]()


class LOG(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return np.log(input_buffer[0]())


class POW(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return np.power(input_buffer[0](), input_buffer[1]())


class SQRT(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return np.sqrt(input_buffer[0]())


class MATMUL(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        # TODO
        pass


class DOT(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return np.dot(input_buffer[0](), input_buffer[1]())


class INNER(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return np.inner(input_buffer[0](), input_buffer[1]())


class OUTER(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return np.outer(input_buffer[0](), input_buffer[1]())


class TENSORDOT(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return np.tensordot(input_buffer[0](), input_buffer[1](), input_buffer[2]())


class KRON(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CHOLESKY(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class QR(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = ''

    def set_mode(self, mode):
        self.mode = mode


class SVD(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_dict = {'full_matrices': 1, 'compute_uv': 1, 'hermitian': 0}

    def set_param(self, full_matrices, compute_uv, hermitian):
        self.parameter_dict['full_matrices'] = full_matrices
        self.parameter_dict['compute_uv'] = compute_uv
        self.parameter_dict['hermitian'] = hermitian


class NORM(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_dict = {'ord': None, 'axis': None, 'keepdims': 0}

    def set_param(self, ord, axis, keepdims):
        self.parameter_dict['ord'] = ord
        self.parameter_dict['axis'] = axis
        self.parameter_dict['keepdims'] = keepdims


class COND(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_dict = {'p': None}

    def set_param(self, p):
        self.parameter_dict['p'] = p


class DET(Node):
    def __init__(self, **kwargs):
        super().__init__(30, **kwargs)


class RANK(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TRACE(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_dict = {'offset': 0, 'axis1': 0, 'axis2': 1, 'dtype': None, 'out': None}

    def set_param(self, offset, axis1, axis2, dtype, out):
        self.parameter_dict['offset'] = offset
        self.parameter_dict['axis1'] = axis1
        self.parameter_dict['axis2'] = axis2
        self.parameter_dict['dtype'] = dtype
        self.parameter_dict['out'] = out


class RESHAPE(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_dict = {'newshape': None, 'order': 'C'}

    def set_param(self, newshape, order):
        self.parameter_dict['newshape'] = newshape
        self.parameter_dict['order'] = order


class TRANSPOSE(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class STACK(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.axis = 0

    def set_axis(self, axis):
        self.axis = axis


# 该类实例含义为当前位置值未知，占空，之后被其他类实例取代
class Blank(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# 该类为列表切片、索引，self.name为列表名，self.slice_info为切片信息
class Slice(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
                total_slice.append(slice(*list(map(lambda x:None if x=='' else int(x), idx.split(':')))))
            else:
                total_slice.append(int(idx))
        self.slice_index = total_slice

    @batch_stream
    @operator_wrapper
    def run(self, input_buffer):
        return input_buffer[0]().__getitem__(self.slice_index)



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
@ shallow_copy
def InstantiationClass(nodeId, nodeType, branches=None, with_grad=False, **otherField):
    if nodeType == 'CreateTensor':
        data_shape = otherField['data_shape']
        node = globals()[nodeType](data_shape, id=nodeId, branches=branches, with_grad=with_grad)
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
    elif nodeType == 'Symbol':
        value = otherField['value']
        node = globals()[nodeType](value, id=nodeId, branches=branches, with_grad=with_grad)
    else:
        node = globals()[nodeType](id=nodeId, branches=branches, with_grad=with_grad)
    return node
