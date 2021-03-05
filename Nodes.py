class Node:
    # 计算图中节点类的父类
    def __init__(self, type_id, physic_algorithm='relational', **kwargs):
        self.physic_algorithm = physic_algorithm
        self.id = kwargs['id']
        self.type_id = type_id
        self.with_grad = kwargs['with_grad']

    def GetId(self):
        return self.id

    def GetType(self):
        return self.type_id

# 通过继承实现的其它节点类
class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)

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

class Val(Node):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)
        self.value = 0

    def set_val(self, value):
        self.value = value

class Sql(Node):
    def __init__(self, t_name, **kwargs):
        super().__init__(3, **kwargs)
        self.t_name = t_name

class Random(Node):
    def __init__(self, uLimit, lLimit, **kwargs):
        super().__init__(4, **kwargs)
        self.uLimit = uLimit
        self.lLimit = lLimit

# 算术表达式所用节点
class Symbol(Node):
    def __init__(self, value, **kwargs):
        super().__init__(5, **kwargs)
        self.value = value

# 逻辑控制所用节点
class Loop(Node):
    def __init__(self,times,loop_id,**kwargs):
        super().__init__(6,**kwargs)
        self.times = times
        self.loop_id = loop_id

class Loop_End(Node):
    def __init__(self,loop_id,**kwargs):
        super().__init__(7,**kwargs)
        self.loop_id = loop_id

class Break(Node):
    def __init__(self,loop_id,**kwargs):
        super().__init__(8,**kwargs)
        self.loop_id = loop_id

class If(Node):
    def __init__(self,**kwargs):
        super().__init__(9,**kwargs)

class If_Branch(Node):
    def __init__(self,**kwargs):
        super().__init__(10,**kwargs)

class If_End(Node):
    def __init__(self,**kwargs):
        super().__init__(11,**kwargs)

class Assignment(Node):
    def __init__(self,**kwargs):
        super().__init__(12,**kwargs)


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
