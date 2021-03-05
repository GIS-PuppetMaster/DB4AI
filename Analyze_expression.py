import re
import Nodes as nd
import Digraph
import math

'''
该文件中解析部分主要分为两部分，
第一部分是Pretreatment()，对给定表达式进行初步解析，提取定义变量等必要部分，
在第一部分中，输入用户DEF定义表达式，输出对表达式初步划分(列表形式)和求导信息
第二部分是Analyze_expression()，考虑到诸多情况，对表达式进行进一步如计算顺序上的划分
在第二部分中，输入第一部分结果和节点初始ID如0，输出图G，变量列表(包括表达式中出现的变量和其对应序号），图G顶端的最上层顶点
表达式输入要按照除算子自带括号外所有符号和字母彼此间由单个空格隔开的格式，如:
    DEF A = B + C * D
    DEF X = Y + LOG(Z)
    DEF M = N * POW(J , K) WITH GRAD
    DEF A = ( B + C ) * D WITH GRAD

'''

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
建立用来解析表达式的图，支持插入(生成子节点），查询子节点，和修改节点类型
'''

'''
在表达式中，给定DEF X = Y + ...，
我们无法确定它是DEF X = Y + PI 还是 DEF X = Y + LOG(..)
因此这里暂定未解析到的节点均为算子Symbol节点，出现变量节点后对该图使用setVal函数进行修改
'''


class build_Graph:
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
            child = build_Graph(node_id, nodeType, value=value)
        else:
            child = build_Graph(node_id, nodeType)
        self.children.append(child)

    def getChild(self):
        return self.children[-1]

    def setVal(self, obj):
        self.keynode = obj


'''
要对输入字符串进行预处理
'''


def Pretreatment(expression):
    requires_grad = False
    use_reg = '.+(WITH GRAD)?'
    if re.match(use_reg, expression) and re.search('WITH GRAD', expression):
        requires_grad = True
    exp_list = expression.split('=')
    Y = exp_list[1]
    if len(exp_list[1].strip()) != 1:
        Y = '( ' + exp_list[1] + ' )'
    return Y.split(), requires_grad


'''
算术表达式解析，之前应进行预处理，
给定字符串s，
Analyze_expression(s)输出图G，
增设了统一算子符号类Symbol
'''

'''
关于变量如何确定变量名和变量值
在算数表达式中，如给定DEF X = Y + Z，
在这个式子我们无法确定 Y 值和 Z 值，
因此在val类node节点中将val值设为 Y 和 Z
可以通过node.value或指定函数访问，
至于变量 Y 和 Z 的具体取值需要比对之前的变量表进行确定
'''


def Analyze_expression(result, x):
    # 划分了单元算子和多元算子
    single_operator = ('LOG', 'SQRT', 'CHOLESKY', 'DET', 'RANK', 'TRANSPOSE')
    multiple_operator = ('POW', 'MATMUL', 'DOT', 'COND', 'INNER', 'OUTER', 'TENSORDOT', 'RACE', 'KRON', 'QR', 'NORM',
                         'RESHAPE', 'STACK')

    # 这里要记录求导信息，设requires_grad变量，此变量应记录在Node父类节点中，之后配合解析器修改
    expression = result[0]
    requires_grad = result[1]
    if requires_grad is True and len(expression) >= 3:
        if expression[-3] == 'WITH' and expression[-2] == 'GRAD':
            expression.pop(-3)
            expression.pop(-2)
    # print(expression)
    for i in expression:

        # 将分散为多个字符串的算子内部表达式整合为一个字符串
        begin = expression.index(i)
        end = 0
        if i.startswith(single_operator) or i.startswith(multiple_operator):
            flag = 1
            count = begin + 1
            while count < len(expression):
                if expression[count].startswith(single_operator) or expression[count].startswith(multiple_operator) \
                        or expression[count].startswith('('):
                    flag += 1
                if expression[count].endswith(')'):
                    flag -= 1
                if flag == 0:
                    end = count
                    break
                count += 1
            t = begin + 1
            s = i
            while t <= end:
                s = s + ' ' + expression[t]
                t += 1
            expression.insert(begin, s)
            count = 1
            while count <= end - begin + 1:
                expression.pop(begin + 1)
                count += 1

    # 对优先级较高的算式添加括号
    new_expression = expression
    count = 0
    while count < len(expression):
        if expression[count] in ['*', '/']:
            if expression[count - 1] != ')' and expression[count + 1] != '(':
                expression.insert(count - 1, '(')
                expression.insert(count + 3, ')')
            count += 1
        count += 1

    count = 0
    expression = new_expression

    # 如果产生多重括号，去重
    while count < len(expression):
        if expression[count] == '(' and expression[count + 1] == '(':
            new_count = count
            new_count += 2
            while expression[new_count] != ')':
                new_count += 1
                continue
            if expression[new_count + 1] == ')':
                expression.pop(count + 1)
                expression.pop(new_count)
        count += 1

    # 初始化
    newStack = Stack()
    G = Digraph.Graph()
    newGraph = build_Graph(x + len(G.nodes), 'Symbol', with_grad=requires_grad, value='')
    newStack.push(newGraph)
    currentgraph = newGraph
    vallist = []

    # 对表达式进行处理
    for i in expression:
        if i == '(':
            currentgraph.insert(x + len(G.nodes), 'Symbol', with_grad=requires_grad, value='')
            newStack.push(currentgraph)
            currentgraph = currentgraph.getChild()
        elif i in ['+', '-', '*', '/']:
            currentgraph.setVal(nd.InstantiationClass(x + len(G.nodes), 'Symbol', with_grad=requires_grad, value=i))
            if len(newStack.items) != 0:
                parent = newStack.pop()
                if currentgraph != parent and parent.keynode.value != '':
                    G.InsertEdge(currentgraph.keynode, parent.keynode)
                newStack.push(parent)
            G.InsertNode(currentgraph.keynode)
            G.InsertEdge(currentgraph.getChild().keynode, currentgraph.keynode)
            currentgraph.insert(len(G.nodes), 'Symbol', with_grad=requires_grad, value='')
            newStack.push(currentgraph)
            currentgraph = currentgraph.getChild()
        elif i == ')':
            currentgraph = newStack.pop()
        elif i == 'PI':
            currentgraph.setVal(nd.InstantiationClass(x + len(G.nodes), 'CreatTensor', with_grad=requires_grad,
                                                      data_shape='(' + str(math.pi) + ')'))
            parent = newStack.pop()
            G.InsertNode(currentgraph.keynode)
            if currentgraph != parent:
                G.InsertEdge(currentgraph.keynode, parent.keynode)
            currentgraph = parent
        elif i == 'E':
            currentgraph.setVal(nd.InstantiationClass(x + len(G.nodes), 'CreatTensor', with_grad=requires_grad,
                                                      data_shape='(' + str(math.e) + ')'))
            parent = newStack.pop()
            G.InsertNode(currentgraph.keynode)
            if currentgraph != parent:
                G.InsertEdge(currentgraph.keynode, parent.keynode)
            currentgraph = parent
        elif i.startswith(single_operator):
            for j in single_operator:
                if i.startswith(j):
                    currentgraph.setVal(nd.InstantiationClass(x + len(G.nodes), 'Symbol',
                                                              value=j, with_grad=requires_grad))
            G.InsertNode(currentgraph.keynode)
            parent = newStack.pop()
            G.InsertEdge(currentgraph.keynode, parent.keynode)
            pattern = re.compile(r'[(](.*?)[)]', re.S)
            variable = re.findall(pattern, i)[0]
            if len(variable.split()) != 1:
                variable = '( ' + re.findall(pattern, i)[0] + ' )'
            temp = Analyze_expression((variable.split(), requires_grad), x + len(G.nodes))
            for k in temp[0].nodes:
                flag = 0
                for e in temp[0].edges:
                    if e.GetStart() is k:
                        flag = 1
                if flag == 0:
                    G.InsertEdge(k, currentgraph.keynode)
                if len(temp[0].edges) == 0:
                    G.InsertEdge(k, currentgraph.keynode)
            G.nodes = G.nodes + temp[0].nodes
            G.edges = G.edges + temp[0].edges
            for j in temp[1]:
                vallist.append(j)
        elif i.startswith(multiple_operator):
            for j in multiple_operator:
                if i.startswith(j):
                    currentgraph.setVal(nd.InstantiationClass(x + len(G.nodes), 'Symbol',
                                                              value=j, with_grad=requires_grad))
            G.InsertNode(currentgraph.keynode)
            parent = newStack.pop()
            G.InsertEdge(currentgraph.keynode, parent.keynode)
            pattern = re.compile(r'[(](.*?)[)]', re.S)
            variable = re.findall(pattern, i)[0].split(',')
            for j in variable:
                # print(j)
                if len(j.split()) != 1:
                    j = '( ' + j.strip() + ' )'
                temp = Analyze_expression((j.split(), requires_grad), x + len(G.nodes))
                for k in temp[0].nodes:
                    flag = 0
                    for e in temp[0].edges:
                        if e.GetStart() is k:
                            flag = 1
                        if flag == 0:
                            G.InsertEdge(k, currentgraph.keynode)
                    if len(temp[0].edges) == 0:
                        G.InsertEdge(k, currentgraph.keynode)
                for k in temp[1]:
                    vallist.append(k)
                G.nodes = G.nodes + temp[0].nodes
                G.edges = G.edges + temp[0].edges
        else:
            vallist.append([i, x + len(G.nodes)])
            currentgraph.setVal(nd.InstantiationClass(x + len(G.nodes), 'Val', with_grad=requires_grad))
            currentgraph.keynode.set_val(i)
            parent = newStack.pop()
            G.InsertNode(currentgraph.keynode)
            if currentgraph != parent and parent.keynode.value != '':
                G.InsertEdge(currentgraph.keynode, parent.keynode)
            currentgraph = parent

    # 返回生成解析树上最上层顶点
    for n in G.nodes:
        flag = 0
        for e in G.edges:
            if e.GetStart() is n:
                flag = 1
                break
        if flag == 0:
            top_node = n
            break
    else:
        top_node = None
    return G.GetSet(), vallist, top_node


if __name__ == '__main__':
    s = 'A = ( B + C ) * D WITH GRAD'
    p = Analyze_expression(Pretreatment(s), 0)
    # for i in p[0].nodes:
    #     print(i, i.value)
    #     # print(i)
    # for i in p[0].edges:
    #     print(i, i.start.value, i.end.value)
    #     # print(i)
    print(p[1])
    print(p[2])