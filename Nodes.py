
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

    def __call__(self, executor: Executor):
        pass
    # def GetType(self):
    #     return self.type_id


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
        data_shape = data_shape.replace('(', ',').replace(')', ',')
        data = data_shape.split(',')
        data_list = []
        for d in data:
            if len(d) != 0:
                data_list.append(int(d))
        self.data_shape = tuple(data_list)

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
    def __init__(self, t_name, **kwargs):
        super().__init__(3, **kwargs)
        self.t_name = t_name

    def __call__(self, executor: Executor):
        # TODO:和高斯数据的API
        pass


class Random(Node):
    def __init__(self, uLimit, lLimit, **kwargs):
        super().__init__(4, **kwargs)
        self.uLimit = uLimit
        self.lLimit = lLimit

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
    def __init__(self, times, loop_id, **kwargs):
        super().__init__(6, **kwargs)
        self.times = times
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


class If_Branch(Node):
    def __init__(self, **kwargs):
        super().__init__(10, **kwargs)


class If_End(Node):
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
        t_name = otherField['t_name']
        node = globals()[nodeType](t_name, id=nodeId, with_grad=with_grad)
    elif nodeType == 'Random':
        uLimit = otherField['uLimit']
        lLimit = otherField['lLimit']
        node = globals()[nodeType](uLimit, lLimit, id=nodeId, with_grad=with_grad)
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
