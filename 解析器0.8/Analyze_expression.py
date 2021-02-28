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
建立用来解析表达式的图，支持插入(生成子节点），查询子节点，和修改节点类型
'''

'''
在表达式中，给定DEF X = Y + ...，
我们无法确定它是DEF X = Y + PI 还是 DEF X = Y + LOG(..)
因此这里暂定未解析到的节点均为算子节点，出现变量节点后对该图使用setVal函数进行修改
该图不是最终返回图，所以影响不大
'''
class build_Graph:
    def __init__(self, node_id, nodeType):
        node = nd.InstantiationClass(node_id, nodeType)
        self.keynode = node
        self.children = []

    def insert(self, nodeType, type_id):
        child = build_Graph(nodeType, type_id)
        self.children.append(child)

    def getChild(self):
        return self.children[len(self.children) - 1]

    def setVal(self, obj):
        self.keynode = obj


'''
要对输入字符串进行预处理
'''


def Pretreatment(expression):
    explist = expression.split()
    if explist[0] != 'DEF':
        return

    '''
    这里要记录求导信息，暂设requires_grad变量，此变量应记录在val类Node节点中，之后配合解析器修改
    '''
    if explist[len(explist) - 2] == 'WITH' and explist[len(explist) - 1] == 'GRAD':
        requires_grad = True

    explist = expression.split('=')
    X = explist[0].split()[1]          # X为DEF定义的变量名，应记录在全局变量中，之后配合解析器修改
    print(len(explist[1].strip()))
    Y = explist[1]
    if len(explist[1].strip()) != 1:
        Y = '( ' + explist[1] + ' )'
    return Y.split()


'''
算术表达式解析，之前应进行预处理，
给定字符串s，
Analyze_expression(Pretreatment(s))输出图G，
增设了统一算子符号类uniform
'''

'''
关于变量如何确定变量名和变量值
在算数表达式中，如给定DEF X = Y + Z，
在这个式子我们无法确定 Y 值和 Z 值，
因此在val类node节点中将val值设为 Y 和 Z
可以通过node.value或指定函数访问，
至于变量 Y 和 Z 的具体取值需要比对之前的变量表进行确定
'''


def Analyze_expression(expression):
    newStack = Stack()
    G = Digraph.Graph()
    newGraph = build_Graph(len(G.nodes), 'Uniform')
    G.InsertNode(newGraph.keynode)
    newStack.push(newGraph)
    currentgraph = newGraph
    for i in expression:
        print(i)
        if i == '(':
            currentgraph.insert(len(G.nodes), 'Uniform')
            newStack.push(currentgraph)
            currentgraph = currentgraph.getChild()
        elif i in ['+', '-', '*', '/']:
            currentgraph.insert(len(G.nodes), 'Uniform')
            newStack.push(currentgraph)
            currentgraph = currentgraph.getChild()
        elif i == ')':
            currentgraph = newStack.pop()
        elif i == 'PI':
            currentgraph.setVal(nd.InstantiationClass(len(G.nodes), 'CreatTensor', data_shape='(' + str(math.pi) + ')'))
            parent = newStack.pop()
            G.InsertNode(currentgraph.keynode)
            if currentgraph != parent:
                G.InsertEdge(currentgraph.keynode, parent.keynode)
            currentgraph = parent
        elif i == 'E':
            currentgraph.setVal(nd.InstantiationClass(len(G.nodes), 'CreatTensor', data_shape='(' + str(math.e) + ')'))
            parent = newStack.pop()
            G.InsertNode(currentgraph.keynode)
            if currentgraph != parent:
                G.InsertEdge(currentgraph.keynode, parent.keynode)
            currentgraph = parent
        elif i.startswith('LOG') or i.startswith('SQRT') or i.startswith('CHOLESKY') or i.startswith('DET') \
            or i.startswith('RANK') or i.startswith('TRANSPOSE'):
            currentgraph.insert(len(G.nodes), 'Uniform')
            G.InsertNode(currentgraph.getChild().keynode)
            G.InsertEdge(currentgraph.getChild().keynode, currentgraph.keynode)
            currentgraph = currentgraph.getChild()
            pattern = re.compile(r'[(](.*?)[)]', re.S)
            variable = re.findall(pattern, i)[0].split()
            if len(variable.strip()) != 1:
                variable = '( ' + re.findall(pattern, i)[0].split() + ' )'
            temp = Analyze_expression(variable.split())
            G.InsertEdge(temp.nodes[0], currentgraph.keynode)
        elif i.startswith('POW') or i.startswith('MATMUL') or i.startswith('DOT')  or i.startswith('COND')\
            or i.startswith('INNER') or i.startswith('OUTER') or i.startswith('TENSORDOT')  or i.startswith('RACE')\
            or i.startswith('KRON') or i.startswith('QR') or i.startswith('NORM') or i.startswith('RESHAPE')\
            or i.startswith('STACK'):
            currentgraph.insert(len(G.nodes), 'Uniform')
            G.InsertNode(currentgraph.getChild().keynode)
            G.InsertEdge(currentgraph.getChild().keynode, currentgraph.keynode)
            currentgraph = currentgraph.getChild()
            pattern = re.compile(r'[(](.*?)[)]', re.S)
            variable = re.findall(pattern, i)[0].split(',')
            for j in variable:
                temp = Analyze_expression(j)
                G.InsertEdge(temp.nodes[0], currentgraph.keynode)
        else:
            currentgraph.setVal(nd.InstantiationClass(len(G.nodes), 'Val'))
            currentgraph.keynode.set_val(i)
            parent = newStack.pop()
            G.InsertNode(currentgraph.keynode)
            if currentgraph != parent:
                G.InsertEdge(currentgraph.keynode, parent.keynode)
            currentgraph = parent
    return G