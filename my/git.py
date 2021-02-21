# from enum import Enum
import re
import json
# 正则识别格式
variable_name_reg = '^[a-zA-Z_][a-zA-Z0-9_]*$'
# TODO: not finished
number_reg = '^[1-9]\d*(\.\d)+'
# 变量名列表
reg_dict={
    f'^CREAT TENSOR {variable_name_reg} (from \d)+(from )'
}
class Graph:
    def __init__(self):
        self.nodes = [Root()]
        self.edges = [[]]


class Node:
    # 计算图中结点的父类
    def __init__(self, type_id, physic_algorithm='relational', **kwargs):
        self.physic_algorithm = physic_algorithm
        self.id = kwargs['id']
        self.type_id = type_id


class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)


class CreatTensor(Node):
    def __init__(self, data_shape, **kwargs):
        super().__init__(1, **kwargs)
        self.data_shape = json.loads(data_shape) # ？？？？
        assert isinstance(self.data_shape.type, tuple)


class Val(Node):
    def __init__(self, value, **kwargs):
        super().__init__(2, **kwargs)
        self.value = value


class Plus(Node):
    def __init__(self, **kwargs):
        super().__init__(3, **kwargs)


class Subtract(Node):
    def __init__(self, **kwargs):
        super().__init__(4, **kwargs)


class Multiply(Node):
    def __init__(self, **kwargs):
        super().__init__(5, **kwargs)


class Divide(Node):
    def __init__(self, **kwargs):
        super().__init__(6, **kwargs)


class Edge:
    def __init__(self):
        self.data_shape = None
        self.data_type = None
        self.data_physic_type = None
        # 表示边上数据在end节点执行函数中的参数位置，
        # 例如2^3，end节点为pow，则2对应的边的parameter_index=1，3对应的边的parameter_index=2
        # 例如3^2，end节点为pow，则3对应的边的parameter_index=1，2对应的边的parameter_index=2
        self.parameter_index = None


class Parser:
    def __init__(self, queries:list):
        self.queries = queries
        self.var_dict = dict()
        self.exp_list = dict()
        self.graph = Graph()

    def __call__(self, *args, **kwargs):
        for query in self.queries:
            pass

