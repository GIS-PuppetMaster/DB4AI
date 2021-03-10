import re
import Nodes as nd
import Digraph
import math


'''
栈用来存储父节点，以支持算子作为父节点联接多个变量子节点
'''


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()


'''
建立用来辅助解析表达式的图，支持插入(生成子节点），查询子节点，和修改节点类型
'''


class BuildGraph:
    def __init__(self, node_id, nodeType, **otherField):
        if nodeType == 'Symbol':
            value = otherField['value']
            node = nd.InstantiationClass(node_id, nodeType, value=value)
        else:
            node = nd.InstantiationClass(node_id, nodeType)
        self.keynode = node
        self.children = []

    def insert(self, node_id, nodeType, **otherField):
        if nodeType == 'Symbol':
            value = otherField['value']
            child = BuildGraph(node_id, nodeType, value=value)
        else:
            child = BuildGraph(node_id, nodeType)
        self.children.append(child)

    def get_child(self):
        return self.children[- 1]

    def get_children(self):
        return self.children

    def set_val(self, obj):
        self.keynode = obj


'''

analyze_expression()负责处理输入和输出

输入包括输入语句和所期望的节点初始序号，如0, 1

其中输入语句是符合规定的DEF表达式语句，即输入中所有符号彼此间由单个空格隔开，括号、参数、矩阵内部符号和切片符号除外，示例如下:

DEF A = B + C * D
DEF X = Y + LOG(Z + Q)
DEF M = N * POW(J , 3) WITH GRAD
DEF A = (B + C) * D WITH GRAD
DEF M = N * TENSORDOT(J , F , ([1,0],[0,1]) WITH GRAD
DEF A = B / NORM(c , ord=1 , axis=0) WITH GRAD
DEF A = a[4:-3,5:-7]

输出图G，变量列表(包括表达式中出现的变量和其生成Val节点对应序号），图G顶端的最上层顶点；

图G叶节点全部为张量或张量切片，非叶节点全部为算子，叶节点通过非叶节点相连，张量因此可以通过连接彼此的算子进行计算

在该函数中，第一步是对给定表达式进行初步解析，提取定义变量，提取表达式'='后内容，记录求导信息

第二步是对表达式进行分割，对算子进行组合并确定算子优先级顺序

第三步是具体解析包括生成节点和图，处理节点参数等

关于节点的参数选择和输入，我们将每类节点提供参数提供如下：

LOG : base
QR : mode
SVD : full_matrices, compute_uv, hermitian
NORM : ord, axis, keepdims
COND : p
TRACE : offset, axis1, axis2, dtype, out
RESHAPE : order
TENSORDOT : axes
STACK : axis

其中节点存在多个参数时输入参数需要提供参数名，如： DEF A = B / NORM(C , ord=1 , axis=0) WITH GRAD
当某一参数输入值包括多种类型，如COND类中p参数可以为int值或"inf"值等，统一按字符串存储
'''


def analyze_expression(expression, x):

    simple_operator = ('+', '-', '*', '/')
    # 在高级算子中划分单元算子(单个变量，不包括属性值）和多元算子
    single_operator = ('LOG', 'POW', 'SQRT', 'CHOLESKY', 'QR', 'SVD', 'NORM', 'COND', 'DET', 'RANK', 'TRACE', 'RESHAPE',
                       'TRANSPOSE', 'GRADIENT')
    multiple_operator = ('MATMUL', 'DOT', 'INNER', 'OUTER', 'TENSORDOT', 'KRON', 'STACK')

    # 这里要记录求导信息，设requires_grad变量，此变量应记录在Node父类节点中，之后配合解析器修改
    requires_grad = False
    use_reg = '.+(WITH GRAD)?'
    if re.match(use_reg, expression) and re.search('WITH GRAD', expression):
        requires_grad = True
    explist = expression.split('=', 1)
    X = explist[0].split()[1]          # X为DEF定义的变量名，应记录在全局变量中，之后配合解析器修改
    Y = explist[1]
    if len(explist[1].strip()) != 1:
        Y = '( ' + explist[1] + ' )'
    expression = Y.split()
    if requires_grad is True and len(expression) >= 3:
        if expression[- 3] == 'WITH' and expression[- 2] == 'GRAD':
            expression.pop(- 3)
            expression.pop(- 2)

    new_expression = []
    for i in expression:
        if i.startswith('(') and len(i) != 1:
            new_expression.append('(')
            new_expression.append(i[1:])
        elif i.endswith(')') and len(i) != 1:
            new_expression.append(i[0:-1])
            new_expression.append(')')
        else:
            new_expression.append(i)
    expression = new_expression

    for i in expression:

        # 将分散为多个字符串的算子内部表达式整合为一个字符串
        begin = expression.index(i)
        end = 0
        if i.startswith(single_operator) or i.startswith(multiple_operator):
            flag = 1
            count = begin + 1
            while count < len(expression):
                flag += expression[count].count('(')
                flag -= expression[count].count(')')
                if flag == 0:
                    end = count
                    break
                count += 1
            new_begin = begin + 1
            s = i
            while new_begin <= end:
                s = s + ' ' + expression[new_begin]
                new_begin += 1
            expression.insert(begin, s)
            count = 1
            while count <= end - begin + 1:
                expression.pop(begin + 1)
                count += 1

    # 对优先级较高的算式添加括号
    # new_expression = expression
    count = 0
    label = 0
    for e in expression:
        if e in ['+', '-']:
            label = 1
    while count < len(expression) and label == 1:
        if expression[count] in ['*', '/']:
            if expression[count - 1] != ')' and expression[count + 1] != '(':
                expression.insert(count - 1, '(')
                expression.insert(count + 3, ')')
            count += 1
        count += 1
    count = 0

    # 如果产生多重括号，去重
    while count + 3 < len(expression):
        if expression[count] == '(' and expression[count + 1] == '(':
            new_count = count
            new_count += 2
            while expression[new_count] != ')':
                new_count += 1
                if new_count < len(expression):
                    continue
            if new_count + 1 < len(expression):
                if expression[new_count + 1] == ')':
                    expression.pop(count + 1)
                    expression.pop(new_count)
        count += 1
    # 初始化
    new_stack = Stack()
    G = Digraph.Graph()
    new_graph = BuildGraph(x, 'Blank', with_grad=requires_grad)
    x += 1
    new_stack.push(new_graph)
    current_graph = new_graph
    vallist = []

    # 对表达式进行处理
    for i in expression:

        # 对当前节点添加子节点，操作节点转移到子节点
        if i == '(':
            current_graph.insert(x, 'Blank', with_grad=requires_grad)
            x += 1
            new_stack.push(current_graph)
            current_graph = current_graph.get_child()

        # 操作节点转移到父节点
        elif i == ')':
            current_graph = new_stack.pop()

        # 设置当前节点值，将当前节点与可能邻接边加入图G，添加子节点，操作节点转移到子节点
        elif i in simple_operator:
            count = 0
            label = 1
            while count < len(simple_operator):
                if count == 0 and i == simple_operator[count]:
                    if current_graph.keynode.GetType() == 36:
                        current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), 'Add', with_grad=requires_grad))
                    else:
                        label = 0
                        temp_graph = BuildGraph(x, 'Add', with_grad=requires_grad)
                        x += 1
                        temp_graph.get_children().append(current_graph)
                        current_graph = temp_graph
                if count == 1 and i == simple_operator[count]:
                    if current_graph.keynode.GetType() == 36:
                        current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), 'Sub', with_grad=requires_grad))
                    else:
                        label = 0
                        temp_graph = BuildGraph(x, 'Sub', with_grad=requires_grad)
                        x += 1
                        temp_graph.get_children().append(current_graph)
                        current_graph = temp_graph
                if count == 2 and i == simple_operator[count]:
                    if current_graph.keynode.GetType() == 36:
                        current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), 'Mul', with_grad=requires_grad))
                    else:
                        label = 0
                        temp_graph = BuildGraph(x, 'Mul', with_grad=requires_grad)
                        x += 1
                        temp_graph.get_children().append(current_graph)
                        current_graph = temp_graph
                if count == 3 and i == simple_operator[count]:
                    if current_graph.keynode.GetType() == 36:
                        current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), 'Div', with_grad=requires_grad))
                    else:
                        label = 0
                        temp_graph = BuildGraph(x, 'Div', with_grad=requires_grad)
                        x += 1
                        temp_graph.get_children().append(current_graph)
                        current_graph = temp_graph
                count += 1
            if len(new_stack.items) != 0 and label == 1:
                parent = new_stack.pop()
                if current_graph != parent and parent.keynode.GetType() != 36:
                    G.InsertEdge(current_graph.keynode, parent.keynode)
                new_stack.push(parent)
            G.InsertNode(current_graph.keynode)
            G.InsertEdge(current_graph.get_child().keynode, current_graph.keynode)

            current_graph.insert(x, 'Blank', grad=requires_grad)
            x += 1
            new_stack.push(current_graph)
            current_graph = current_graph.get_child()

        # 对于constant.PI和constant.E，节点值为对应张量，操作节点转移到父节点
        elif i == 'PI':
            current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), 'Val', value=math.pi, with_grad=requires_grad))
            current_graph.keynode.set_val(math.pi)
            parent = new_stack.pop()
            G.InsertNode(current_graph.keynode)
            if current_graph != parent and parent.keynode.GetType() != 36:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            current_graph = parent
        elif i == 'E':
            current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), 'Val', value=math.e, with_grad=requires_grad))
            current_graph.keynode.set_val(math.e)
            parent = new_stack.pop()
            G.InsertNode(current_graph.keynode)
            if current_graph != parent and parent.keynode.GetType() != 36:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            current_graph = parent

        # 对于算子，设置当前节点值，识别自带括号内内容，新建子图G’，连接G和G'
        elif i.startswith(single_operator):
            for j in single_operator:
                if i.startswith(j):
                    current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), j, with_grad=requires_grad))
                    break
            G.InsertNode(current_graph.keynode)
            parent = new_stack.pop()
            if current_graph != parent and parent.keynode.GetType() != 36:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            pattern = re.compile(r'[(](.*?)[)]', re.S)
            variable = re.findall(pattern, i)[0].split(',')
            new_expression = 'DEF ' + X + ' = ' + variable[0]
            if requires_grad:
                new_expression = new_expression + ' WITH GRAD'
            temp = analyze_expression(new_expression, x)
            # 对单变量进行解析操作
            for k in temp[0][0]:
                flag = 0
                for e in temp[0][1]:
                    if e.GetStart() is k:
                        flag = 1
                        break
                if flag == 0:
                    G.InsertEdge(k, current_graph.keynode)
            G.nodes = G.nodes + temp[0][0]
            G.edges = G.edges + temp[0][1]
            for t in temp[1]:
                vallist.append(t)

            # 对参数进行解析操作
            if j == 'POW':
                base = variable[1]
                if variable[1].find('base') != -1:
                    base = variable[1].split('=')[1].strip()
                current_graph.keynode.set_base(base)
            if j == 'QR':
                mode = variable[1]
                if variable[1].find('mode') != -1:
                    mode = variable[1].split('=')[1].strip()
                current_graph.keynode.set_base(mode)
            if j == 'SVD':
                count = 1
                full_matrices = 1
                compute_uv = 1
                hermitian = 0
                while count < len(variable):
                    if variable[count].find('full_matrices') != -1:
                        full_matrices = eval(variable[count].split('=')[1].strip())
                        print(variable[count], full_matrices)
                    if variable[count].find('compute_uv') != -1:
                        compute_uv = eval(variable[count].split('=')[1].strip())
                    if variable[count].find('hermitian') != -1:
                        hermitian = eval(variable[count].split('=')[1].strip())
                    count += 1
                current_graph.keynode.set_param(full_matrices, compute_uv, hermitian)
            if j == 'NORM':
                count = 1
                ord = None
                axis = None
                keepdims = 0
                while count < len(variable):
                    if variable[count].find('ord') != -1:
                        ord = eval(variable[count].split('=')[1].strip())
                    if variable[count].find('axis') != -1:
                        axis = eval(variable[count].split('=')[1].strip())
                    if variable[count].find('keepdims') != -1:
                        keepdims = eval(variable[count].split('=')[1].strip())
                    count += 1
                current_graph.keynode.set_param(ord, axis, keepdims)
            if j == 'COND':
                count = 1
                p = variable[count].strip()
                while count < len(variable):
                    if variable[count].find('p') != -1:
                        p = variable[count].split('=')[1].strip()
                    count += 1
                current_graph.keynode.set_param(p)
            if j == 'TRACE':
                count = 1
                offset = 0
                axis1 = 0
                axis2 = 1
                dtype = None
                out = None
                while count < len(variable):
                    if variable[count].find('offset') != -1:
                        offset = eval(variable[count].split('=')[1].strip())
                    if variable[count].find('axis1') != -1:
                        axis1 = eval(variable[count].split('=')[1].strip())
                    if variable[count].find('axis2') != -1:
                        axis2 = eval(variable[count].split('=')[1].strip())
                    if variable[count].find('dtype') != -1:
                        dtype = eval(variable[count].split('=')[1].strip())
                    if variable[count].find('out') != -1:
                        out = eval(variable[count].split('=')[1].strip())
                    count += 1
                current_graph.keynode.set_param(offset, axis1, axis2, dtype, out)
            if j == 'RESHAPE':
                count = 2
                newshape = variable[1]
                order = 'C'
                while count < len(variable):
                    order = variable[count]
                    if variable[count].find('order') != -1:
                        order = variable[count].split(',')[1].strip()
                    count += 1
                current_graph.keynode.set_param(newshape, order)

        elif i.startswith(multiple_operator):
            for j in multiple_operator:
                if i.startswith(j):
                    current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), j, with_grad=requires_grad))
                    break
            G.InsertNode(current_graph.keynode)
            parent = new_stack.pop()
            if current_graph != parent and parent.keynode.GetType() != 36:
                G.InsertEdge(current_graph.keynode, parent.keynode)

            # 关于后续是否直接输入张量
            if j == 'TENSORDOT':
                variable = i[len(j) + 1:-1].strip().split(',', 2)
            else:
                variable = i[len(j) + 1:-1].strip().split(',', 1)

            for v in variable:

                if j == 'TENSORDOT' and variable.index(v) == 2:
                    axes = v.strip()
                    if axes.startswith('axes'):
                        axes = v.split('=')[1].strip()
                    current_graph.keynode.set_axes(axes)
                    break

                if j == 'STACK' and variable.index(v) >= 1:
                    axis = v.strip()
                    if v.find('axis') != -1:
                        axis = v.split('=')[1].strip()
                    current_graph.keynode.set_axis(axis)
                    break

                new_expression = 'DEF ' + X + ' = ' + v.strip()
                if requires_grad:
                    new_expression = new_expression + ' WITH GRAD'
                temp = analyze_expression(new_expression, x)
                x += len(temp[0][0])

                for k in temp[0][0]:
                    flag = 0
                    for e in temp[0][1]:
                        if e.GetStart() is k:
                            flag = 1
                            break
                    if flag == 0:
                        G.InsertEdge(k, current_graph.keynode)
                for k in temp[1]:
                    vallist.append(k)
                G.nodes = G.nodes + temp[0][0]
                G.edges = G.edges + temp[0][1]

        # 识别列表切片
        elif re.search(re.compile(r'\[(.*?)\]', re.S), i):
            current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), 'Slice', with_grad=requires_grad))
            current_graph.keynode.set_name(i[:i.index('[')])
            slice_info = re.findall(re.compile(r'\[(.*?)\]', re.S), i)
            new_slice_info = []
            for s in slice_info[0].split(','):
                new_s = s.strip()
                new_slice_info.append(new_s)
            current_graph.keynode.set_slice(new_slice_info)
            parent = new_stack.pop()
            G.InsertNode(current_graph.keynode)
            if current_graph != parent and parent.keynode.GetType() != 36:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            current_graph = parent

        # 对于未识别字符设定为变量，设置当前节点值，将当前节点与可能邻接边加入图G，操作节点转移到父节点
        else:
            vallist.append([i, x])
            current_graph.set_val(nd.InstantiationClass(current_graph.keynode.GetId(), 'Val', with_grad=requires_grad))
            current_graph.keynode.set_val(i)
            parent = new_stack.pop()
            G.InsertNode(current_graph.keynode)
            if current_graph != parent and parent.keynode.GetType() != 36:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            current_graph = parent

    # 返回生成解析树上最上层顶点
    for n in G.nodes:
        flag = 0
        for e in G.edges:
            if e.GetStart() == n.GetId():
                flag = 1
                break
        if flag == 0:
            top_node = n
            break
    return G.GetSet(), vallist, top_node


if __name__ == '__main__':
    s = "DEF a = x + y / z - x * x + POW(T , 3)"
    # s = "DEF s = N * TENSORDOT(J , F , ([1,0],[0,1])"
    p = analyze_expression(s, 0)
    for i in p[0][1]:
        print(i.GetStart(), i.GetEnd())
    print(p[1])
