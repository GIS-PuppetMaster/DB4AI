import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

import Nodes as nd
import Digraph
import numpy

'''
栈用来存储父节点，以支持算子作为父节点联接多个变量子节点
'''
variable_name_reg = '([a-zA-Z]+[a-zA-Z0-9_]*)'


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        try:
            return self.items.pop()
        except(KeyError, IndexError):
            print("栈为空，无法pop")


'''
建立用来辅助解析表达式的图，支持插入(生成子节点），查询子节点，和修改节点类型
'''


class BuildGraph:
    def __init__(self, node_id, nodeType, branches, **otherField):
        node = nd.InstantiationClass(node_id, nodeType, branches, **otherField)
        self.keynode = node
        self.children = []

    def insert(self, node_id, nodeType, branches, **otherField):
        child = BuildGraph(node_id, nodeType, branches, **otherField)
        self.children.append(child)

    def get_child(self):
        try:
            return self.children[-1]
        except IndexError:
            raise Exception('无子节点')

    def get_children(self):
        return self.children

    def set_val(self, obj):
        self.keynode = obj


'''
analyze_expression()负责处理输入和输出
输入包括输入语句和所期望的节点初始序号，如0, 1，以及所在分支branches列表和replace集合
其中输入语句是DEF表达式语句，示例如下:
A=B+C*D
X=Y+LOG(Z+Q)
M = N * POW(J,3) WITH GRAD
A = (B+C)*D WITH GRAD
M = N * TENSORDOT(J, F, ([1,0],[0,1]) WITH GRAD
A = B/NORM(c,ord=1,axis=0) WITH GRAD
A = a[4:-3,5:-7]
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
其中节点存在多个参数时输入参数需要提供参数名，如：A = B/NORM(C,ord=1,axis=0) WITH GRAD
当某一参数输入值包括多种类型，如COND类中p参数可以为int值或"inf"值等，统一按字符串存储
'''

simple_operator = ('+', '-', '*', '/')
# 在高级算子中划分单元算子(单个变量，不包括属性值）和多元算子以及零元算子
none_operator = ('Ones', 'Zeros')
single_operator = ('LOG', 'POW', 'SQRT', 'CHOLESKY', 'QR', 'SVD', 'NORM', 'COND', 'DET', 'RANK', 'TRACE', 'RESHAPE',
                   'TRANSPOSE', 'SHAPE', 'EXP', 'Deepcopy', 'Shallowcopy', 'Argmax', 'Argmin', 'Sign', 'SaveTable',
                   'SUM', 'Relu', 'Tanh', 'Softmax', 'Sigmod', 'Elu', 'MEAN', 'MAX', 'MIN', 'Abs', 'ARGSORT', 'SORT',
                   'REVERSE', 'GRADIENT', 'UNSQUEEZE', 'TensorFromSql')
multiple_operator = ('MATMUL', 'DOT', 'INNER', 'OUTER', 'TENSORDOT', 'KRON', 'STACK', 'Adam', 'AUC', 'MSE',
                     'F1', 'ACC', 'RECALL', 'PRECISION', 'WLS', 'REPEAT', 'Backward', 'CleanGrad')
all_operator = nd.operators


def analyze_expression(expression, x, inner_count, branches: list, replace=None):
    if replace is None:
        replace = {}

    # 常量dict,用于建立对应val节点
    constant_dict = {'CONSTANT.E': numpy.e, 'CONSTANT.PI': numpy.pi}

    # 查看UserOperators.json文件，取得自定义算子
    if os.path.exists('UserOperatorName.json'):
        with open('UserOperatorName.json', 'r') as f:
            load_dict = json.load(f, strict=False)
            # load_dict = json.load(f)
            user_operator = load_dict.get('name')
        user_operator = tuple(user_operator)
    else:
        user_operator = ()

    # 这里要记录求导信息，设requires_grad变量，此变量应记录在Node父类节点中，之后配合解析器修改
    requires_grad = False
    use_reg = '.+(WITH GRAD)?'
    if re.match(use_reg, expression) and re.search('WITH GRAD', expression):
        requires_grad = True
    explist = expression.split('=', 1)
    val_name = explist[0]  # X为DEF定义的变量名，应记录在全局变量中，之后配合解析器修改
    y = explist[1].strip()

    # 去除多余空格
    new_y = ''
    for i in range(len(y)):
        if y[i] != ' ':
            new_y += y[i]
    if requires_grad is True:
        new_y = new_y[:new_y.find('WITH')] + ' ' + new_y[new_y.find('WITH'):]
        new_y = new_y[:new_y.find('GRAD')] + ' ' + new_y[new_y.find('GRAD'):]
    y = new_y

    new_exp = ''
    for i in range(len(y)):
        new_i = y[i]
        if y[i] in simple_operator:

            # 避免在负数加入空格
            if y[i] == '-' and re.fullmatch(re.compile('\\d'), y[i + 1]) and (y[i - 1] in (':', '+', '-', '*', '/', '(')
                                                                              or i == 0):
                pass

            elif y[i - 1] != ' ' or y[i + 1] != ' ':
                if y[i - 1] != ' ':
                    new_i = ' ' + new_i
                if y[i + 1] != ' ':
                    new_i = new_i + ' '
            new_exp = new_exp + new_i
        else:
            new_exp = new_exp + new_i

    # 对表达式首尾加括号
    expression = new_exp.split()
    if len(expression) != 1:
        expression.insert(0, '(')
        expression.append(')')

    # 去除WITH GRAD标志位
    if requires_grad is True and len(expression) >= 3:
        if expression[- 3] == 'WITH' and expression[- 2] == 'GRAD':
            expression.pop(- 3)
            expression.pop(- 2)

    # 检测是否存在不分隔括号，形如'(a','bc)'
    flag = 1
    while flag == 1:
        flag = 0
        new_expression = []
        for i in expression:
            if i.startswith('(') and len(i) != 1:
                new_expression.append('(')
                new_expression.append(i[1:])
                flag = 1
            elif i.endswith(')') and len(i) != 1:
                new_expression.append(i[0:-1])
                new_expression.append(')')
                flag = 1
            else:
                new_expression.append(i)
        expression = new_expression

    # 替换replace中需替换变量
    for i in expression:
        if len(replace) == 0:
            break
        if replace.get(i) is not None:
            expression[expression.index(i)] = replace.get(i)

    # 将分散为多个字符串的算子内部表达式整合为一个字符串
    for i in expression:
        begin = expression.index(i)
        end = 0
        if i.startswith(single_operator) or i.startswith(multiple_operator) or i.startswith(user_operator) \
                or i.startswith(none_operator):
            flag = 0
            count = begin
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

    # 如果产生多重括号，去重
    count = 0
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

    # 对优先级较高的算式添加括号
    count = 0
    label0 = 0
    label1 = 0
    for e in expression:
        if e in ['+', '-']:
            label0 = 1
        if e in ['*', '/'] and label0 == 1:
            label1 = 1
    while count < len(expression) and label1 == 1:
        if expression[count] in ['*', '/']:
            flag = 0
            front = count - 1
            cnt = 0
            while front >= 0:
                if expression[front] == '(':
                    cnt += 1
                if expression[front] == ')':
                    cnt -= 1
                front -= 1
            if cnt > 1:
                flag = 1
            cnt = 0
            if flag == 0:
                if expression[count - 1] != ')':
                    expression.insert(count - 1, '(')
                    count += 1
                else:
                    cnt = 1
                    t = count - 2
                    while t >= 0:
                        if expression[t] == ')':
                            cnt += 1
                        if expression[t] == '(':
                            cnt -= 1
                        if cnt == 0:
                            break
                        t -= 1
                    expression.insert(t, '(')
                    count += 1
                if expression[count + 1] != '(':
                    expression.insert(count + 2, ')')
                    count += 1
                else:
                    cnt = 1
                    t = count + 2
                    while t < len(expression):
                        if expression[t] == '(':
                            cnt += 1
                        if expression[t] == ')':
                            cnt -= 1
                        if cnt == 0:
                            break
                        t += 1
                    expression.insert(t, ')')
                    count += 1
        count += 1

    # 初始化图等信息
    new_stack = Stack()
    G = Digraph.Graph()
    new_graph = BuildGraph(x, 'Blank', branches, with_grad=requires_grad)
    new_stack.push(new_graph)
    current_graph = new_graph
    vallist = []

    # 对表达式进行处理
    for i in expression:

        # 对当前节点添加子节点，操作节点转移到子节点
        if i == '(':
            current_graph.insert(x, 'Blank', branches, with_grad=requires_grad)
            new_stack.push(current_graph)
            current_graph = current_graph.get_child()

        # 操作节点转移到父节点
        elif i == ')':
            parent = new_stack.pop()
            flag = 0
            for e in G.edges:
                if e.GetStart() == current_graph.keynode and e.GetEnd() == parent.keynode:
                    flag = 1
                if e.GetStart() == parent.keynode and e.GetEnd() == current_graph.keynode:
                    flag = 1
            if flag == 0 and current_graph.keynode != parent.keynode and not isinstance(parent.keynode, nd.Blank) \
                    and parent.keynode in G.without_out:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            current_graph = parent

        # 设置当前节点值，将当前节点与可能邻接边加入图G，添加子节点，操作节点转移到子节点
        elif i in simple_operator:
            flag = 0
            if i == '-':
                if expression.index(i) == 1:
                    current_graph.set_val(nd.InstantiationClass(x, 'Negative', branches, with_grad=requires_grad))
                    x += 1
                    flag = 1
                elif expression[expression.index(i) - 1] == '(' or expression[expression.index(i) - 1] in simple_operator:
                    current_graph.set_val(nd.InstantiationClass(x, 'Negative', branches, with_grad=requires_grad))
                    x += 1
                    flag = 1
            if flag == 0:
                simple_operator_class = ['Add', 'Sub', 'Mul', 'Div']
                for count in range(len(simple_operator)):
                    if i == simple_operator[count]:
                        if isinstance(current_graph.keynode, nd.Blank):
                            current_graph.set_val(
                                nd.InstantiationClass(x, simple_operator_class[count], branches, with_grad=requires_grad))
                            x += 1
                        else:
                            label = 0
                            parent_graph = BuildGraph(x, simple_operator_class[count], branches, with_grad=requires_grad)
                            x += 1
                            parent_graph.get_children().append(current_graph)
                            current_graph = parent_graph
            G.InsertNode(current_graph.keynode)
            try:
                if len(current_graph.get_children()) != 0:
                    G.InsertEdge(current_graph.get_child().keynode, current_graph.keynode)
            except AttributeError:
                raise Exception('无子图')
            current_graph.insert(x, 'Blank', branches, grad=requires_grad)
            new_stack.push(current_graph)
            current_graph = current_graph.get_child()

        # 对于constant.PI和constant.E，节点值为对应张量，操作节点转移到父节点
        elif i in constant_dict.keys():
            current_graph.set_val(nd.InstantiationClass(current_graph.keynode.id, 'Val', branches,
                                                        val=constant_dict.get(i), with_grad=requires_grad))
            x += 1
            parent = new_stack.pop()
            G.InsertNode(current_graph.keynode)
            if current_graph != parent and isinstance(parent.keynode, nd.Blank) is not True:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            vallist.append([i, current_graph.keynode])
            current_graph = parent

        # 对于算子，设置当前节点值，识别自带括号内内容，新建子图G’，连接G和G'
        elif i.startswith(single_operator):
            for j in single_operator:
                if i.startswith(j):
                    current_graph.set_val(
                        nd.InstantiationClass(current_graph.keynode.id, j, branches, with_grad=requires_grad))
                    x += 1
                    break
            j = j.strip()
            G.InsertNode(current_graph.keynode)
            parent = new_stack.pop()
            if current_graph != parent and isinstance(parent.keynode, nd.Blank) is not True:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            new_stack.push(parent)
            temp_i = i[len(j) + 1:-1].strip()
            cnt = 0
            begin = 0
            var = []
            for l in range(len(temp_i)):
                if temp_i[l] in ['(', '[']:
                    cnt += 1
                if temp_i[l] in [')', ']']:
                    cnt -= 1
                if temp_i[l] == ',' and cnt == 0:
                    end = l
                    var.append(temp_i[begin:end])
                    begin = l + 1
                if l == len(temp_i) - 1:
                    var.append(temp_i[begin:])
            if len(var) == 0:
                var.append(temp_i)
            if j == 'SaveTable':
                current_graph.keynode.set_name(var[1])
                if var[2] == "print":
                    current_graph.keynode.print_flag = True
                current_graph.keynode.set_vars([None, var[0]])
                input_node = nd.InstantiationClass(x, 'Var', branches, vars=var[0], with_grad=requires_grad)
                G.InsertNode(input_node)
                G.InsertEdge(input_node, current_graph.keynode)
                vallist.append([var[0], input_node])
                current_graph = new_stack.pop()
                continue
            new_expression = val_name + '=' + var[0]
            if requires_grad:
                new_expression = new_expression + ' WITH GRAD'
            temp = analyze_expression(new_expression, x, inner_count, branches, replace)
            x += len(temp[0][0])
            for k in temp[0][0]:
                G.InsertNode(k)
                G.without_out.remove(k)

            # 对单变量进行解析操作,找到顶点
            for k in temp[0][0]:
                flag = 0
                for e in temp[0][1]:
                    if e.GetStart() is k:
                        flag = 1
                        break
                if flag == 0:
                    G.InsertEdge(k, current_graph.keynode)
            for k in temp[0][1]:
                G.edges.append(k)
            for t in temp[1]:
                vallist.append(t)

            # 对参数进行解析操作
            if j == 'POW':
                exp = var[1]
                if re.fullmatch(re.compile('[-]?\d+'), exp.strip()):
                    exp_node = nd.InstantiationClass(x, 'Val', branches, val=eval(exp), with_grad=requires_grad)
                    exp_node.set_val(eval(exp))
                    x += 1
                    G.InsertNode(exp_node)
                    G.InsertEdge(exp_node, current_graph.keynode)
                    vallist.append([exp.strip(), exp_node])
                else:
                    exp_expression = val_name + '=' + exp
                    if requires_grad:
                        exp_expression = exp_expression + 'WITH GRAD'
                    temp = analyze_expression(exp_expression, x, inner_count, branches, replace)
                    for k in temp[0][0]:
                        G.InsertNode(k)
                        G.without_out.remove(k)
                    for k in temp[0][0]:
                        flag = 0
                        for e in temp[0][1]:
                            if e.GetStart() is k:
                                flag = 1
                                break
                        if flag == 0:
                            G.InsertEdge(k, current_graph.keynode)
                    for k in temp[0][1]:
                        G.edges.append(k)
                    for t in temp[1]:
                        vallist.append(t)
            elif j == 'QR':
                mode = var[1]
                if var[1].find('mode') != -1:
                    mode = var[1].split('=')[1].strip()
                current_graph.keynode.set_mode(mode)
            elif j == 'SVD':
                count = 1
                full_matrices = 1
                compute_uv = 1
                hermitian = 0
                while count < len(var):
                    if var[count].find('full_matrices') != -1:
                        full_matrices = eval(var[count].split('=')[1].strip())
                        print(var[count], full_matrices)
                    if var[count].find('compute_uv') != -1:
                        compute_uv = eval(var[count].split('=')[1].strip())
                    if var[count].find('hermitian') != -1:
                        hermitian = eval(var[count].split('=')[1].strip())
                    count += 1
                current_graph.keynode.set_param(full_matrices, compute_uv, hermitian)
            elif j == 'NORM':
                count = 1
                ord = None
                axis = None
                keepdims = 0
                while count < len(var):
                    if var[count].find('ord') != -1:
                        ord = eval(var[count].split('=')[1].strip())
                    if var[count].find('axis') != -1:
                        axis = eval(var[count].split('=')[1].strip())
                    if var[count].find('keepdims') != -1:
                        keepdims = eval(var[count].split('=')[1].strip())
                    count += 1
                current_graph.keynode.set_param(ord, axis, keepdims)
            elif j == 'COND':
                count = 1
                p = var[count].strip()
                while count < len(var):
                    if var[count].find('p') != -1:
                        p = var[count].split('=')[1].strip()
                    count += 1
                current_graph.keynode.set_param(p)
            elif j == 'TRACE':
                count = 1
                offset = 0
                axis1 = 0
                axis2 = 1
                dtype = None
                out = None
                while count < len(var):
                    if var[count].find('offset') != -1:
                        offset = eval(var[count].split('=')[1].strip())
                    if var[count].find('axis1') != -1:
                        axis1 = eval(var[count].split('=')[1].strip())
                    if var[count].find('axis2') != -1:
                        axis2 = eval(var[count].split('=')[1].strip())
                    if var[count].find('dtype') != -1:
                        dtype = eval(var[count].split('=')[1].strip())
                    if var[count].find('out') != -1:
                        out = eval(var[count].split('=')[1].strip())
                    count += 1
                current_graph.keynode.set_param(offset, axis1, axis2, dtype, out)
            elif j == 'RESHAPE':
                count = 2
                newshape = var[1]
                order = 'C'
                while count < len(var):
                    order = var[count]
                    if var[count].find('order') != -1:
                        order = var[count].split('=')[1].strip()
                    count += 1
                current_graph.keynode.set_param(newshape, order)
            elif j == 'SUM':
                if len(var) != 1:
                    if type(var[1]) == str:
                        current_graph.keynode.set_axis(eval(var[1]))
                    else:
                        current_graph.keynode.set_axis(var[1])
            elif j == 'Elu':
                current_graph.keynode.set_alpha(var[1])
            elif j == 'MEAN':
                if len(var) != 1:
                    if type(var[1]) == str:
                        current_graph.keynode.set_axis(eval(var[1]))
                    else:
                        current_graph.keynode.set_axis(var[1])
            elif j in ['Argmax', 'Argmin', 'MAX', 'MIN']:
                if len(var) != 1:
                    if type(var[1]) == str:
                        current_graph.keynode.set_axis(eval(var[1]))
                    else:
                        current_graph.keynode.set_axis(var[1])
            elif j == 'UNSQUEEZE':
                if len(var) != 1:
                    if type(var[1]) == str:
                        current_graph.keynode.set_dim(eval(var[1]))
                    else:
                        current_graph.keynode.set_dim(var[1])
            elif j == 'Softmax':
                if len(var) != 1:
                    if type(var[1]) == str:
                        current_graph.keynode.set_dim(eval(var[1]))
                    else:
                        current_graph.keynode.set_dim(var[1])
            current_graph = new_stack.pop()
        elif i.startswith(multiple_operator):
            for j in multiple_operator:
                if i.startswith(j):
                    current_graph.set_val(
                        nd.InstantiationClass(current_graph.keynode.id, j, branches, with_grad=requires_grad))
                    x += 1
                    break
            G.InsertNode(current_graph.keynode)
            parent = new_stack.pop()
            if current_graph != parent and isinstance(parent.keynode, nd.Blank) is not True:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            new_stack.push(parent)

            # Adam算子目前可选参数仅支持learning_rate，单独处理
            if j == 'Adam':
                var = i[len(j) + 1:-1].strip().split(',')
                for v in var:
                    v = v.strip()
                    if v.startswith('learning_rate'):
                        current_graph.keynode.set_learning_rate(v.split('=')[1])
                    else:
                        input_node = nd.InstantiationClass(x, 'Var', branches, vars=v, with_grad=requires_grad)
                        x += 1
                        G.InsertNode(input_node)
                        G.InsertEdge(input_node, current_graph.keynode)
                        vallist.append([v, input_node])
                current_graph = new_stack.pop()
                continue
            # 分离参数
            temp_i = i[len(j) + 1:-1].strip()
            cnt = 0
            begin = 0
            var = []
            for l in range(len(temp_i)):
                if temp_i[l] in ['(', '[']:
                    cnt += 1
                if temp_i[l] in [')', ']']:
                    cnt -= 1
                if temp_i[l] == ',' and cnt == 0:
                    end = l
                    var.append(temp_i[begin:end])
                    begin = l + 1
                if l == len(temp_i) - 1:
                    var.append(temp_i[begin:])
            if len(var) == 0:
                var.append(temp_i)
            # 遍历参数
            for v in var:
                if j == 'TENSORDOT' and var.index(v) == 2:
                    axes = var[1]
                    if re.fullmatch(re.compile('\d+'), axes.strip()):
                        axes_node = nd.InstantiationClass(x, 'Val', branches, val=eval(axes), with_grad=requires_grad)
                        x += 1
                        G.InsertNode(axes_node)
                        G.InsertEdge(axes_node, current_graph.keynode)
                    else:
                        axes_node = nd.InstantiationClass(x, 'Var', branches, vars=axes.strip(),
                                                          with_grad=requires_grad)
                        x += 1
                        G.InsertNode(axes_node)
                        G.InsertEdge(axes_node, current_graph.keynode)
                    break
                if j == 'STACK' and var.index(v) >= 1:
                    axis = v.strip()
                    if v.find('axis') != -1:
                        axis = v.split('=')[1].strip()
                    current_graph.keynode.set_axis(axis)
                    break
                new_expression = val_name + '=' + v.strip()
                if requires_grad:
                    new_expression = new_expression + 'WITH GRAD'
                temp = analyze_expression(new_expression, x, inner_count, branches, replace)
                x += len(temp[0][0])
                for k in temp[0][0]:
                    G.InsertNode(k)
                    G.without_out.remove(k)
                for k in temp[0][0]:
                    flag = 0
                    for e in temp[0][1]:
                        if e.GetStart() is k:
                            flag = 1
                            break
                    if flag == 0:
                        G.InsertEdge(k, current_graph.keynode)
                        break
                for k in temp[1]:
                    vallist.append(k)
                for k in temp[0][1]:
                    G.edges.append(k)
            current_graph = new_stack.pop()

        # 该部分支持直接生成张量算子，目前有Ones和Zeros
        elif i.startswith(none_operator):
            for j in none_operator:
                if i.startswith(j):
                    params = i[len(j) + 1:-1]
                    pattern = re.compile(r'[(](.*?)[)]', re.S)
                    if re.search(pattern, params):
                        data_shape = re.search(pattern, params)
                        print(data_shape)
                        print(data_shape.group(0))
                        new_params = params[data_shape.end(0) + 1:-1]
                        param = new_params.strip().split(',')
                        current_graph.set_val(nd.InstantiationClass(current_graph.keynode.id, j, branches,
                                                                    with_grad=requires_grad,
                                                                    data_shape=data_shape.group(0),
                                                                    var=param[0].strip()))
                    else:
                        param = params.split(',')
                        current_graph.set_val(nd.InstantiationClass(current_graph.keynode.id, j, branches,
                                                                    with_grad=requires_grad,
                                                                    data_shape=param[0].strip(),
                                                                    var=param[1].strip()))
                    x += 1
                    break
            parent = new_stack.pop()
            G.InsertNode(current_graph.keynode)
            if current_graph != parent and isinstance(parent.keynode, nd.Blank) is not True:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            current_graph = parent

        # 自定义算子，通过访问SecondLevelLanguageParser.py文件生成的UserOperatorName.json以及UserOperatorInfo
        # 获取文件名和对应自定义算子内容，即输出、输入和图，该部分在SecondLevelLanguageParser.py的AddUserOperator函数中实现
        elif i.startswith(user_operator):
            for j in user_operator:
                if i.startswith(j):
                    with open('UserOperatorInfo', 'rb') as f:
                        t = pickle.load(f)
                        operator_info = t.get(j)
                    break
            # operator_info[2].Show()
            formal_param = []
            for o in operator_info[1]:
                formal_param.append(o[0])
            operator_info[2].ChangeNodeInfo(len(G.nodes) - len(operator_info[1]) + x, branches, with_grad=requires_grad)
            for node in operator_info[2].nodes:
                for t in range(len(node.vars)):
                    if t == 0 and node.vars[t] is None:
                        continue
                    if not node.vars[t].startswith('_') and node.vars[t] not in formal_param:
                        node.vars[t] = '__' + str(inner_count) + node.vars[t]
                if hasattr(node, 'data_shape'):
                    pattern = re.compile(r'[(](.*?)[)]', re.S)
                    try:
                        data_shape = re.findall(pattern, str(node.data_shape))[0].split(',')
                    except TypeError:
                        print(node.data_shape)
                        print(type(node.data_shape))
                    for p in range(len(data_shape)):
                        if not data_shape[p].strip().isdigit() and not data_shape[p].startswith('_') and data_shape[p] not in formal_param and data_shape[p]!='':
                            data_shape[p] = '__' + str(inner_count) + data_shape[p]
                    str_data_shape = '('
                    for p in data_shape:
                        str_data_shape = str_data_shape + p + ','
                    str_data_shape = str_data_shape[:-1] + ')'
                    node.data_shape = str_data_shape
                if hasattr(node, 'data_shape_var'):
                    for key in list(node.data_shape_var.keys()):
                        if not key.startswith('_') and key not in formal_param:
                            node.data_shape_var['__' + str(inner_count) + key] = node.data_shape_var.pop(key)
                if hasattr(node, 'boundary_var'):
                    for key in list(node.boundary_var.keys()):
                        if not key.startswith('_') and key not in formal_param:
                            node.boundary_var['__' + str(inner_count) + key] = node.boundary_var.pop(key)
                if hasattr(node, 'dead_cycle'):
                    if isinstance(node.dead_cycle, str):
                        if not node.dead_cycle.startswith('_') and node.dead_cycle not in formal_param:
                            node.dead_cycle = '__' + str(inner_count) + node.dead_cycle
                if hasattr(node, '_slice'):
                    if node.slice is not None:
                        for p in range(len(node.slice)):
                            if isinstance(node.slice[p], str):
                                if not node.slice[p].startswith('_') and node.slice[p] not in formal_param:
                                    node.slice[p] = '__' + str(inner_count) + node.slice[p]
                            if isinstance(node.slice[p], slice):
                                new_start = node.slice[p].start
                                new_stop = node.slice[p].stop
                                new_step = node.slice[p].step
                                if isinstance(node.slice[p].start, str):
                                    if not node.slice[p].start.startswith('_') and node.slice[p].start not in formal_param:
                                        new_start = '__' + str(inner_count) + node.slice[p].start
                                if isinstance(node.slice[p].stop, str):
                                    if not node.slice[p].stop.startswith('_') and node.slice[p].stop not in formal_param:
                                        new_stop = '__' + str(inner_count) + node.slice[p].stop
                                if isinstance(node.slice[p].step, str):
                                    if not node.slice[p].step.startswith('_') and node.slice[p].step not in formal_param:
                                        new_step = '__' + str(inner_count) + node.slice[p].step
                                node.slice[p] = slice(new_start, new_stop, new_step)
                if hasattr(node, 'slice_index'):
                    if node.slice_index is not None:
                        for p in range(len(node.slice_index)):
                            if isinstance(node.slice_index[p], str):
                                if not node.slice_index[p].startswith('_') and node.slice_index[p] not in formal_param:
                                    node.slice_index[p] = '__' + str(inner_count) + node.slice_index[p]
                            if isinstance(node.slice_index[p], slice):
                                new_start = node.slice_index[p].start
                                new_stop = node.slice_index[p].stop
                                new_step = node.slice_index[p].step
                                if isinstance(node.slice_index[p].start, str):
                                    if not node.slice_index[p].start.startswith('_') and node.slice_index[p].start not in formal_param:
                                        new_start = '__' + str(inner_count) + node.slice_index[p].start
                                if isinstance(node.slice_index[p].stop, str):
                                    if not node.slice_index[p].stop.startswith('_') and node.slice_index[p].stop not in formal_param:
                                        new_stop = '__' + str(inner_count) + node.slice_index[p].stop
                                if isinstance(node.slice_index[p].step, str):
                                    if not node.slice_index[p].step.startswith('_') and node.slice_index[p].step not in formal_param:
                                        new_step = '__' + str(inner_count) + node.slice_index[p].step
                                node.slice_index[p] = slice(new_start, new_stop, new_step)
            # 预备topnode
            # operator_info[2].Show()
            G.without_in = G.without_in | operator_info[2].without_in
            G.without_out = G.without_out | operator_info[2].without_out
            parent = new_stack.pop()
            if not isinstance(parent.keynode, nd.Blank):
                for r in range(len(list(operator_info[0]))):
                    G.InsertEdge(list(operator_info[0])[r], parent.keynode)
            pattern = re.compile(r'[(](.*?)[)]', re.S)
            var = re.findall(pattern, i)[0].split(',')
            for v in range(len(var)):
                var[v] = var[v].strip()
            # 将算子输入的实际参数变量加入表达式出现的变量列表
            for e in operator_info[2].edges:
                # 如果形参出现
                for inp in operator_info[1]:
                    if e.GetStart() in inp:
                        # 如果变量列表中事先未出现与形参对应的实参(如定义形式为first(a,...)，对应实际调用为first(x,...),则a与x相对应)，
                        # 则将实参加入变量列表
                        if [var[operator_info[1].index(inp)].strip(), e.GetEnd()] not in vallist:
                            vallist.append([var[operator_info[1].index(inp)], e.GetEnd()])
            # 将vars里的输入参数替换
            for n in operator_info[2].nodes:
                for v in range(len(n.vars)):
                    for inp in operator_info[1]:
                        if n.vars[v] == inp[0]:
                            n.vars[v] = var[operator_info[1].index(inp)]
                if hasattr(n, 'data_shape'):
                    pattern = re.compile(r'[(](.*?)[)]', re.S)
                    data_shape = re.findall(pattern, n.data_shape)[0].split(',')
                    for p in range(len(data_shape)):
                        for inp in operator_info[1]:
                            if data_shape[p] == inp[0]:
                                data_shape[p] = var[operator_info[1].index(inp)]
                    str_data_shape = '('
                    for p in data_shape:
                        str_data_shape = str_data_shape + p + ','
                    str_data_shape = str_data_shape[:-1] + ')'
                    n.data_shape = str_data_shape
                if hasattr(n, 'data_shape_var'):
                    for key in list(n.data_shape_var.keys()):
                        for inp in operator_info[1]:
                            if key == inp[0]:
                                n.data_shape_var[var[operator_info[1].index(inp)]] = n.data_shape_var.pop(key)
            if_out_edges = defaultdict(dict)
            for n in range(len(operator_info[2].nodes) - len(operator_info[1])):
                node = operator_info[2].nodes[len(operator_info[1]) + n]
                if isinstance(node, nd.If):
                    for edge in node.out_edges:
                        if_out_edges[node][edge.end] = edge
                G.nodes.append(operator_info[2].nodes[len(operator_info[1]) + n])

            x += len(operator_info[2].nodes) - len(operator_info[1])
            # 遍历图中每条边，符合要求的添加到图G中
            for e in operator_info[2].edges:
                flag = 0
                for input in operator_info[1]:
                    # 与输入参数比对成功，去除与输入参数相连的那条边
                    if e.GetStart() == input[1]:
                        flag = 1
                        for in_edge in e.GetEnd().in_edges:
                            if in_edge.GetStart() == input[1]:
                                e.GetEnd().in_edges.remove(in_edge)
                for t in range(len(e.GetStart().vars)):
                    if isinstance(e.GetStart(), nd.Val) and not e.GetStart().vars[t].startswith('__'):
                        old_var = e.GetStart().vars[t]
                        e.GetStart().vars[t] = '__' + e.GetStart().vars[t][1:]
                        for tt in range(len(e.GetEnd().vars)):
                            if e.GetEnd().vars[tt] == old_var:
                                e.GetEnd().vars[tt] = '__' + e.GetEnd().vars[tt][1:]
                # 若不是形参，则添加到图G中
                if flag == 0:
                    G.edges.append(e)
                    if isinstance(e.GetStart(), nd.If):
                        old_edge = if_out_edges[e.GetStart()][e.GetEnd()]
                        G.edges[-1].condition = old_edge.condition
                        G.edges[-1].reverse = old_edge.reverse
                        G.edges[-1].need_var = old_edge.need_var
            for o in range(len(operator_info[0])):
                if isinstance(list(operator_info[0])[o], nd.Val) or list(operator_info[0])[o].__class__.__name__ in all_operator:
                    list(operator_info[0])[o].set_vars('_' + str(list(operator_info[0])[o].id))
            inner_count += 1
            current_graph.set_val(list(operator_info[0])[0])
            current_graph = parent

        # 识别列表切片
        elif re.search(re.compile(r'\[(.*?)\]', re.S), i):
            current_graph.set_val(
                nd.InstantiationClass(current_graph.keynode.id, 'Slice', branches, with_grad=requires_grad))
            x += 1
            a = nd.InstantiationClass(x, 'Var', branches, vars=i[:i.index('[')], with_grad=requires_grad)
            G.InsertNode(a)
            G.InsertEdge(a, current_graph.keynode)
            vallist.append([i[:i.index('[')], a])
            x += 1
            slice_info = i[i.index('[') + 1:i.rfind(']')]
            new_slice_info = []
            if slice_info.find(',') == -1:
                new_slice_info.append(slice_info)
            else:
                for s in slice_info.split(','):
                    new_slice_info.append(s.strip())
            current_graph.keynode.set_slice(new_slice_info)
            parent = new_stack.pop()
            G.InsertNode(current_graph.keynode)
            if current_graph != parent and isinstance(parent.keynode, nd.Blank) is not True:
                G.InsertEdge(current_graph.keynode, parent.keynode)

            current_graph = parent
            # 处理切片中的变量
            for part in new_slice_info:
                part_list = part.split(':')
                for symbol in part_list:
                    match_obj = re.match(f'{variable_name_reg}[ \t]*', symbol)
                    if match_obj:
                        for obj in match_obj.groups():
                            a = nd.InstantiationClass(x, 'Var', branches, vars=obj, with_grad=False)
                            G.InsertNode(a)
                            G.InsertEdge(a, current_graph.keynode)
                            vallist.append([obj,a])

        # 若未识别字符为数字，则识别为常量，否则设定为变量，设置当前节点值，将当前节点与可能邻接边加入图G，操作节点转移到父节点
        else:
            if re.fullmatch(re.compile('[-]?\\d+([\\.]\\d+)?'), i):
                current_graph.set_val(nd.InstantiationClass(current_graph.keynode.id, 'Val', branches, val=eval(i),
                                                            with_grad=requires_grad))
                x += 1
            else:
                current_graph.set_val(
                    nd.InstantiationClass(current_graph.keynode.id, 'Var', branches, vars=i, with_grad=requires_grad))
                x += 1
            vallist.append([i, current_graph.keynode])
            parent = new_stack.pop()
            G.InsertNode(current_graph.keynode)
            if current_graph != parent and isinstance(parent.keynode, nd.Blank) is not True:
                G.InsertEdge(current_graph.keynode, parent.keynode)
            current_graph = parent

    # 返回生成解析树上最上层顶点
    top_node = None
    try:
        top_node = list(G.GetNoOutNodes())[0]
    except IndexError:
        print("无topnode")

    # 对算子节点添加输入输出信息
    if isinstance(top_node, nd.Val) or top_node.__class__.__name__ in all_operator:
        if not isinstance(top_node, nd.Backward) and not isinstance(top_node, nd.Assignment) and not isinstance(top_node, nd.CleanGrad):
            top_node.set_vars('_' + str(top_node.id))
    for e in G.edges:
        if isinstance(e.GetStart(), nd.Val) or e.GetStart().__class__.__name__ in all_operator:
            if not isinstance(e.GetStart(), nd.Backward):
                if len(e.GetStart().get_vars()) == 0:
                    e.GetStart().set_vars('_' + str(e.GetStart().id))
    for e in G.edges:
        if e.GetEnd().__class__.__name__ in all_operator:
            if len(e.GetStart().get_vars()) != 0 and len(e.GetEnd().get_vars()) - 1 < len(e.GetEnd().in_edges):
                if isinstance(e.GetEnd(), nd.Backward):
                    if len(e.GetEnd().get_vars()) < len(e.GetEnd().in_edges):
                        e.GetEnd().set_vars(e.GetStart().get_vars()[0])
                else:
                    e.GetEnd().set_vars(e.GetStart().get_vars()[0])
    # G.Show()
    return G.GetSet(), vallist, inner_count


if __name__ == '__main__':
    # s = 'y = SUM(n*y*(xa*x),1)+b'
    # s = "loss=y*LOG(hx)+(1-y)*(1-hx)"
    # s = "g=GRADIENT(loss,w)"
    # s = "w=learning_rate*g+w"
    # s = "X =Y+GRADIENT(a,CONSTANT.PI)+3"
    # s = "z = MATMUL(x,w)"
    # s = 's = 1/((c+d)*(e+f))'
    # s = 's =  xx * Zeros(3,7)'
    # s = 's= 5 + TRACE(a,offset=1,axis1=1,axis2=0,dtype=1,out=1) * d'
    # s = 'hx = 1 / (1 + POW(CONSTANT.E, MATMUL(x, w))) WITH GRAD'
    # s = 'loss = y * LOG(hx) + (1 - y) * (1 - hx)'
    # s = 'g = GRADIENT(loss, w)'
    # s = 'w = learning_rate * g + w'
    s = 'y = logistic(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y, ridge, learning_rate, class_num, iter_times)'
    # s = 's = eps*(a2+alpha2+eps*3)'
    s = 's = logistic(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y, ridge, learning_rate, class_num, iter_times)'
    # s = 's = Backward(X,Y,z,n)'
    # s = 's = UNSQUEEZE(x,1)'
    # s = 's = Backward(loss)'
    s = 's = Backward(x,y,loss)'
    s = 's = Softmax(x,1)'
    s = 's = rbf_network(acc,auc,prec,recall,mse,f1, test_x, test_y, train_x, train_y,centers, class_number,learning_rate, batch_size, iter_times)'
    # s = 's = EXP((-beta)*SQRT(SUM(POW(A-B,2),2)))'
    # s = 's = KNNF(acc,auc,prec,recall,mse,f1, data_input, test_y, x, y, k)'
    # s = 's = -3'
    s = 's = a[:,-1]'
    p = analyze_expression(s, 0, 0, [])
    print(p[0][2])
    print(p[1])
    # print(p[2])

    # p[3].Show()
