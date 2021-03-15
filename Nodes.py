from Executor import *


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
        self.vars = []

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

    def GetType(self):
        return self.type_id

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
    # def GetType(self):
    #     return self.type_id


# 通过继承实现的其它节点类
class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# 创建张量所用节点
class CreateTensor(Node):
    def __init__(self, data_shape, **kwargs):
        super().__init__(**kwargs)
        self.data_shape = eval(data_shape)

    def __call__(self, executor: Executor):
        executor.graph.var_dict[self.vars[0]] = Tensor(shape=self.data_shape)

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape
            if self.physic_algorithm == 'relational':
                edge.data_type = 'relation'
            else:
                edge.data_type = 'ndarray'


class Val(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = 0

    def set_val(self, value):
        self.value = value

    def __call__(self, executor: Executor):
        tensor = Tensor(shape=(1,))
        executor.graph.var_dict[self.vars[0]] = tensor.handle = self.value

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = (1,)
            edge.data_type = 'ndarray'


class Sql(Node):
    def __init__(self, t_info, **kwargs):
        super().__init__(**kwargs)
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
        super().__init__(**kwargs)
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
        executor.graph.var_dict[self.vars[0]] = tensor

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_type = 'ndarray'
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
        super().__init__(**kwargs)
        self.loop_id = loop_id

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
        super().__init__(**kwargs)
        self.loop_id = loop_id

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
        super().__init__(**kwargs)

    def next_nodes(self, executor: Executor):
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

    def next_nodes(self, executor: Executor):
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        executor.var_dict[self.vars[0]] = executor.var_dict[self.vars[1]]


class Add(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = executor.var_dict[self.vars[1]].handle + executor.var_dict[self.vars[2]].handle


class Sub(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = executor.var_dict[self.vars[1]].handle - executor.var_dict[self.vars[2]].handle


class Mul(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = executor.var_dict[self.vars[1]].handle * executor.var_dict[self.vars[2]].handle


class Div(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = executor.var_dict[self.vars[1]].handle / executor.var_dict[self.vars[2]].handle


class LOG(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = np.log(executor.var_dict[self.vars[1]].handle)


class POW(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = np.power(executor.var_dict[self.vars[1]].handle, executor.var_dict[self.vars[2]].handle)


class SQRT(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = np.sqrt(executor.var_dict[self.vars[1]].handle)


class MATMUL(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = np.matmul(executor.var_dict[self.vars[1]].handle, executor.var_dict[self.vars[2]].handle)

class DOT(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = np.dot(executor.var_dict[self.vars[1]].handle, executor.var_dict[self.vars[2]].handle)


class INNER(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = np.inner(executor.var_dict[self.vars[1]].handle, executor.var_dict[self.vars[2]].handle)


class OUTER(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = np.outer(executor.var_dict[self.vars[1]].handle, executor.var_dict[self.vars[2]].handle)


class TENSORDOT(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, executor: Executor):
        if self.physic_algorithm == 'relation':
            # TODO
            pass
        else:
            for i in range(1, len(self.vars)):
                temp = executor.var_dict[self.vars[i]]
                if not isinstance(temp, np.ndarray):
                    temp.to_cpu()
            executor.var_dict[self.vars[0]].handle = np.dot(executor.var_dict[self.vars[1]].handle, executor.var_dict[self.vars[2]].handle)


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
        super().__init__(**kwargs)


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
        self.slice_info = []

    def set_name(self, name):
        self.name = name

    def set_slice(self, slice_info):
        self.slice_info += slice_info


# 通过globals方法，以类名选择类进行实例化
def InstantiationClass(nodeId, nodeType, with_grad=False, **otherField):
    if nodeType == 'CreateTensor':
        data_shape = otherField['data_shape']
        node = globals()[nodeType](data_shape, id=nodeId, with_grad=with_grad)
    elif nodeType == 'Sql':
        t_info = otherField['t_info']
        node = globals()[nodeType](t_info, id=nodeId, with_grad=with_grad)
    elif nodeType == 'Random':
        boundary = otherField['boundary']
        data_shape = otherField['data_shape']
        type = otherField['type']
        node = globals()[nodeType](boundary, data_shape, type, id=nodeId, with_grad=with_grad)
    elif nodeType == 'Loop':
        condition = otherField['condition']
        loop_id = otherField['loop_id']
        node = globals()[nodeType](condition, loop_id, id=nodeId, with_grad=with_grad)
    elif nodeType == 'LoopEnd' or nodeType == 'Break':
        loop_id = otherField['loop_id']
        node = globals()[nodeType](loop_id, id=nodeId, with_grad=with_grad)
    elif nodeType == 'Symbol':
        value = otherField['value']
        node = globals()[nodeType](value, id=nodeId, with_grad=with_grad)
    else:
        node = globals()[nodeType](id=nodeId, with_grad=with_grad)
    return node
