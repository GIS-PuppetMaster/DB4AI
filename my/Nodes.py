class Node:
    # 计算图中结点类的父类
    def __init__(self, type_id, physic_algorithm='relational', **kwargs):
        self.physic_algorithm = physic_algorithm
        self.id = kwargs['id']
        self.type_id = type_id
    def GetId(self):
        return self.id
    def GetType(self):
        return self.type_id
# 通过继承实现的其它结点类
class Root(Node):
    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)

# 创建张量所用结点
class CreatTensor(Node):
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
    def __init__(self, value, **kwargs):
        super().__init__(2, **kwargs)
        self.value = value

class Sql(Node):
    def __init__(self,**kwargs):
        super().__init__(3,**kwargs)

class Random(Node):
    def __init__(self,uLimit,lLimit,**kwargs):
        super().__init__(4,**kwargs)
        self.uLimit = uLimit
        self.lLimit = lLimit

# 算术表达式所用结点
class Plus(Node):
    def __init__(self, **kwargs):
        super().__init__(5, **kwargs)

class Subtract(Node):
    def __init__(self, **kwargs):
        super().__init__(6, **kwargs)

class Multiply(Node):
    def __init__(self, **kwargs):
        super().__init__(7, **kwargs)

class Divide(Node):
    def __init__(self, **kwargs):
        super().__init__(8, **kwargs)

# 逻辑控制所用结点
class Loop(Node):
    def __init__(self,**kwargs):
        super().__init__(9,**kwargs)

class Loop_End(Node):
    def __init__(self,**kwargs):
        super().__init__(10,**kwargs)

class G_Relsult(Node):
    def __init__(self,**kwargs):
        super().__init__(11,**kwargs)

class If(Node):
    def __init__(self,**kwargs):
        super().__init__(12,**kwargs)

# 通过globals方法，以类名选择类进行实例化
def InstantiationClass(nodeId,nodeType,**otherField):
    if(nodeType == 'Val'):
        value = otherField['value']
        node = globals()[nodeType](value,id=nodeId)
    elif(nodeType == 'CreatTensor'):
        data_shape = otherField['data_shape']
        node = globals()[nodeType](data_shape,id=nodeId)
    elif(nodeType == 'Random'):
        uLimit = otherField['uLimit']
        lLimit = otherField['lLimit']
        node = globals()[nodeType](uLimit,lLimit,id=nodeId)
    else:
        node = globals()[nodeType](id=nodeId)
    return node
