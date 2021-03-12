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


# 该类用来存储常量，常见如constant.PI、constant.E
class Val(Node):
    def __init__(self, value, **kwargs):
        super().__init__(2, **kwargs)
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


# 逻辑控制所用节点
class Loop(Node):
    def __init__(self, times, loop_id, **kwargs):
        super().__init__(5, **kwargs)
        self.times = times
        self.loop_id = loop_id


class Loop_End(Node):
    def __init__(self, loop_id, **kwargs):
        super().__init__(6, **kwargs)
        self.loop_id = loop_id


class Break(Node):
    def __init__(self, loop_id, **kwargs):
        super().__init__(7, **kwargs)
        self.loop_id = loop_id


class If(Node):
    def __init__(self, **kwargs):
        super().__init__(8, **kwargs)


class If_Branch(Node):
    def __init__(self, **kwargs):
        super().__init__(9, **kwargs)


class If_End(Node):
    def __init__(self, **kwargs):
        super().__init__(10, **kwargs)


class Assignment(Node):
    def __init__(self, **kwargs):
        super().__init__(11, **kwargs)


# 算术表达式所用节点
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
        self.base = 0

    def set_base(self, base):
        self.base = base


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
        self.axes = 2

    def set_axes(self, axes):
        self.axes = axes


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
    elif nodeType == 'Val':
        value = otherField['value']
        node = globals()[nodeType](value, id=nodeId, with_grad=with_grad)
    else:
        node = globals()[nodeType](id=nodeId, with_grad=with_grad)
    return node

