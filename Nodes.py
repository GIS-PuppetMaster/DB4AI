from Executor import *
from copy import copy


class Node:
    # 计算图中节点类的父类
    def __init__(self, type_id, physic_algorithm='relational', **kwargs):
        self.physic_algorithm = physic_algorithm
        self.id = kwargs['id']
        self.type_id = type_id
        self.with_grad = kwargs['with_grad']
        self.out_edges = []
        self.in_edges = []
        self.input_data_edges = []
        self.branches = kwargs['branches']
        self.vars = []

    def generate_data_edges(self):
        for in_edge in self.in_edges:
            if isinstance(in_edge.start, Loop) or isinstance(in_edge.start, If):
                for data_edge in in_edge.start.in_edges:
                    if in_edge.var == data_edge.var:
                        self.input_data_edges.append(data_edge)
            else:
                self.input_data_edges.append(in_edge)

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.id == other.id) and (self.type_id == other.type_id)
        else:
            return False

    def __hash__(self):
        return hash(self.id + self.type_id)

    def __call__(self, executor: Executor):
        pass

    def next_nodes(self, executor: Executor):
        return [edge.end for edge in self.out_edges]

    def infer_data(self):
        pass

    def set_vars(self, input):
        self.vars.append(input)

    def get_vars(self):
        return self.vars


# 通过继承实现的其它节点类
class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)


# 创建张量所用节点
class CreateTensor(Node):
    def __init__(self, data_shape, var,**kwargs):
        super().__init__(1, **kwargs)
        if data_shape:
            self.data_shape = eval(data_shape)
        else:
            self.data_shape = None
        self.var = var

    def __call__(self, executor: Executor):
        executor.graph.output_of_nodes[self] = Tensor(shape=self.data_shape)

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape
            if self.physic_algorithm == 'relational':
                edge.data_type = 'relation'
            else:
                edge.data_type = 'ndarray'

    def get_val(self):
        return self.var

class Val(Node):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)
        self.value = 0

    def set_val(self, value):
        self.value = value

    def get_val(self):
        return self.value

    def __call__(self, executor: Executor):
        tensor = Tensor(shape=(1,))
        executor.graph.output_of_nodes[self] = tensor.handle = self.value

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = (1,)
            edge.data_type = 'ndarray'


class Sql(Node):
    def __init__(self, t_info, **kwargs):
        super().__init__(3, **kwargs)
        self.t_search_sentences = t_info

    def __call__(self, executor: Executor):
        # TODO:和高斯数据的API
        pass

    def infer_data(self):
        for edge in self.out_edges:
            # TODO: 使用SQL查询该表的维度
            # edge.data_shape =
            edge.data_type = 'relation'


class Random(Node):
    def __init__(self, boundary, data_shape, distribution, **kwargs):
        super().__init__(4, **kwargs)
        self.boundary = boundary
        self.data_shape = eval(data_shape)
        if distribution == '':
            self.distribution = 'normal'
        else:
            self.distribution = distribution

    def __call__(self, executor: Executor):
        tensor = Tensor(shape=self.data_shape)
        if self.distribution == 'normal':
            # boundary[0]=lower_boundary, boundary[1]=upper_boundary
            data = np.random.random(self.data_shape) * (self.boundary[1] - self.boundary[0]) + self.boundary[0]
        elif self.distribution == 'gauss':
            # boundary[0]=mu, boundary[1]=sigma
            data = np.random.randn() * self.boundary[1] + self.boundary[0]
        else:
            raise Exception(f'Not supported distribution:{self.distribution}')
        tensor.handle = data
        executor.graph.output_of_nodes[self] = tensor.handle

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

    def __call__(self, executor: Executor):
        executor.output_of_nodes[self] = []
        for edge in self.in_edges:
            start_node = edge.start
            executor.output_of_nodes[self].append(executor.output_of_nodes[start_node])

    def next_nodes(self, executor: Executor):
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
            executor.finished_loop_id.add(self.loop_id)
            return [loop_end_node]
        else:
            return end_nodes


class LoopEnd(Node):
    def __init__(self, loop_id, **kwargs):
        super().__init__(7, **kwargs)
        self.loop_id = loop_id

    def __call__(self, executor: Executor):
        pass

    def next_nodes(self, executor: Executor):
        end_nodes = [edge.end for edge in self.out_edges]
        loop_node = None
        for node in end_nodes:
            if isinstance(node, Loop) and node.loop_id == self.loop_id:
                loop_node = node
                break
        assert loop_node is not None, f'Did not find corresponding loop node for end loop node{self.loop_id}'
        end_nodes.remove(loop_node)
        # 退出循环
        if self.loop_id in executor.finished_loop_id:
            executor.finished_loop_id.remove(self.loop_id)
            return end_nodes
        # 继续下一次循环
        else:
            return [loop_node]


class Break(Node):
    def __init__(self, loop_id, **kwargs):
        super().__init__(8, **kwargs)
        self.loop_id = loop_id

    def __call__(self, executor: Executor):
        pass

    def next_nodes(self, executor: Executor):
        executor.finished_loop_id.remove(self.loop_id)
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

    def __call__(self, executor: Executor):
        pass

    def next_nodes(self, executor: Executor):
        for edge in self.out_edges:
            # todo: edge.node_name
            para = {}
            for var_name, var_node in zip(edge.node_name, edge.node_var):
                para[var_name] = var_node
            res = eval(edge.condition, para)
            if edge.reverse:
                res = not res
            if res:
                return [edge.end]


class IfBranch(Node):
    def __init__(self, **kwargs):
        super().__init__(10, **kwargs)

    def __call__(self, executor: Executor):
        pass

    def next_nodes(self, executor: Executor):
        for edge in self.out_edges:
            # todo: edge.node_name
            para = {}
            for var_name, var_node in zip(edge.node_name, edge.node_var):
                para[var_name] = var_node
            res = eval(edge.condition, para)
            if edge.reverse:
                res = not res
            if res:
                return [edge.end]


class IfEnd(Node):
    def __init__(self, **kwargs):
        super().__init__(11, **kwargs)

    def __call__(self, executor: Executor):
        pass

    def next_nodes(self, executor: Executor):
        return [edge.end for edge in self.out_edges]


class Assignment(Node):
    def __init__(self, var_li, **kwargs):
        super().__init__(12, **kwargs)
        self.var_li = var_li

    def __call__(self, executor: Executor):
        assert len(self.input_data_edges) == 2, f'the number of assignment node\'s in_edges not equal to 2, {self.in_edges}'
        # var_node = self.in_edges[0]
        data_node = self.input_data_edges[1].start
        executor.output_of_nodes[self] = executor.output_of_nodes[data_node]

    def next_nodes(self, executor: Executor):
        return [edge.end for edge in self.out_edges]


class Add(Node):
    def __init__(self, **kwargs):
        super().__init__(12, **kwargs)


class Sub(Node):
    def __init__(self, **kwargs):
        super().__init__(13, **kwargs)


class Mul(Node):
    def __init__(self, **kwargs):
        super().__init__(14, **kwargs)


class Div(Node):
    def __init__(self, **kwargs):
        super().__init__(15, **kwargs)


class LOG(Node):
    def __init__(self, **kwargs):
        super().__init__(16, **kwargs)


class POW(Node):
    def __init__(self, **kwargs):
        super().__init__(17, **kwargs)


class SQRT(Node):
    def __init__(self, **kwargs):
        super().__init__(18, **kwargs)


class MATMUL(Node):
    def __init__(self, **kwargs):
        super().__init__(19, **kwargs)


class DOT(Node):
    def __init__(self, **kwargs):
        super().__init__(20, **kwargs)


class INNER(Node):
    def __init__(self, **kwargs):
        super().__init__(21, **kwargs)


class OUTER(Node):
    def __init__(self, **kwargs):
        super().__init__(22, **kwargs)


class TENSORDOT(Node):
    def __init__(self, **kwargs):
        super().__init__(23, **kwargs)


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
        self.parameter_dict = {'full_matrices': 1, 'compute_uv': 1, 'hermitian': 0}

    def set_param(self, full_matrices, compute_uv, hermitian):
        self.parameter_dict['full_matrices'] = full_matrices
        self.parameter_dict['compute_uv'] = compute_uv
        self.parameter_dict['hermitian'] = hermitian


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


class RANK(Node):
    def __init__(self, **kwargs):
        super().__init__(31, **kwargs)


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


class RESHAPE(Node):
    def __init__(self,  **kwargs):
        super().__init__(33, **kwargs)
        self.parameter_dict = {'newshape': None, 'order': 'C'}

    def set_param(self, newshape, order):
        self.parameter_dict['newshape'] = newshape
        self.parameter_dict['order'] = order


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
        self.slice_info = []

    def set_name(self, name):
        self.name = name

    def set_slice(self, slice_info):
        self.slice_info += slice_info


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
@ shallow_copy
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