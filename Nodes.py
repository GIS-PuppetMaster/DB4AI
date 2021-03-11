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

    def GetId(self):
        return self.id

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


# 通过继承实现的其它节点类
class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)

    def __call__(self, executor: Executor):
        return [edge.end for edge in self.out_edges]


# 创建张量所用节点
class CreateTensor(Node):
    def __init__(self, data_shape, **kwargs):
        super().__init__(1, **kwargs)
        self.data_shape = eval(data_shape)

    def __call__(self, executor: Executor):
        executor.graph.output_of_nodes[self] = Tensor(shape=self.data_shape)
        return [edge.end for edge in self.out_edges]


class Val(Node):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)
        self.value = 0

    def set_val(self, value):
        self.value = value

    def __call__(self, executor: Executor):
        tensor = Tensor(shape=(1,))
        executor.graph.output_of_nodes[self] = tensor.handle = self.value
        return [edge.end for edge in self.out_edges]


class Sql(Node):
    def __init__(self, t_info, **kwargs):
        super().__init__(3, **kwargs)
        search_info = t_info.split('$')
        if search_info[0] == 'W':
            self.column_name = search_info[2]
            self.table_name = search_info[3]
            self.where = search_info[4]
        elif search_info[0] == 'C':
            self.column_name = search_info[1]
            self.table_name = search_info[2]
            self.where = ''
        else:
            self.column_name = ''
            self.table_name = search_info[0]
            self.where = ''

    def __call__(self, executor: Executor):
        # TODO:和高斯数据的API
        pass


class Random(Node):
    def __init__(self, boundary, data_shape, type, **kwargs):
        super().__init__(4,**kwargs)
        self.boundary = boundary
        self.data_shape = eval(data_shape)
        self.type = type

    def __call__(self, executor: Executor):
        # TODO
        # tensor = Tensor(shape=(1,))
        # graph.output_of_nodes[self] = tensor.handle = self.value
        pass


# 算术表达式所用节点
class Symbol(Node):
    def __init__(self, value, **kwargs):
        super().__init__(5, **kwargs)
        self.value = value


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
        # for edge in self.out_edges:
        pass


class IfBranch(Node):
    def __init__(self, **kwargs):
        super().__init__(10, **kwargs)


class IfEnd(Node):
    def __init__(self, **kwargs):
        super().__init__(11, **kwargs)


class Assignment(Node):
    def __init__(self, **kwargs):
        super().__init__(12, **kwargs)


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
