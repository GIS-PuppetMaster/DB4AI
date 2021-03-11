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

    def GetId(self):
        return self.id

    def generate_data_edges(self):
        # TODO: parser change
        for in_edge in self.in_edges:
            if isinstance(in_edge.start, Loop) or isinstance(in_edge.start, If):
                for data_edge in in_edge.start.in_edges:
                    if in_edge.var_name == data_edge.var_name:
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
        super().__init__(0, **kwargs)


# 创建张量所用节点
class CreateTensor(Node):
    def __init__(self, data_shape, **kwargs):
        super().__init__(1, **kwargs)
        self.data_shape = eval(data_shape)

    def __call__(self, executor: Executor):
        executor.graph.output_of_nodes[self] = Tensor(shape=self.data_shape)

    def infer_data(self):
        for edge in self.out_edges:
            edge.data_shape = self.data_shape
            if self.physic_algorithm == 'relational':
                edge.data_type = 'relation'
            else:
                edge.data_type = 'ndarray'


class Val(Node):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)
        self.value = 0

    def set_val(self, value):
        self.value = value

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
        self.t_info = t_info

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


# 算术表达式所用节点
# TODO: 拆分
class Symbol(Node):
    def __init__(self, value, **kwargs):
        super().__init__(5, **kwargs)
        self.value = value


# 逻辑控制所用节点
class Loop(Node):
    def __init__(self, times, loop_id, **kwargs):
        super().__init__(6, **kwargs)
        self.times = times
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
    def __init__(self, **kwargs):
        super().__init__(12, **kwargs)

    def __call__(self, executor: Executor):
        assert len(self.input_data_edges) == 2, f'the number of assignment node\'s in_edges not equal to 2, {self.in_edges}'
        # var_node = self.in_edges[0]
        data_node = self.input_data_edges[1]
        executor.output_of_nodes[self] = executor.output_of_nodes[data_node]

    def next_nodes(self, executor: Executor):
        return [edge.end for edge in self.out_edges]


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
        times = otherField['times']
        loop_id = otherField['loop_id']
        node = globals()[nodeType](times, loop_id, id=nodeId, with_grad=with_grad)
    elif nodeType == 'Loop_End' or nodeType == 'Break':
        loop_id = otherField['loop_id']
        node = globals()[nodeType](loop_id, id=nodeId, with_grad=with_grad)
    elif nodeType == 'Symbol':
        value = otherField['value']
        node = globals()[nodeType](value, id=nodeId, with_grad=with_grad)
    else:
        node = globals()[nodeType](id=nodeId, with_grad=with_grad)
    return node
