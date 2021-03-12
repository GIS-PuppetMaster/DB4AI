import Nodes as nd
import numpy as np
from graphviz import Digraph
from queue import Queue as qu
from Nodes import Node


class Edge:
    def __init__(self, start: Node, end: Node, condition, in_var=None, out_var=None, **kwargs):
        self.data_shape = None
        self.data_type = None
        self.data_physical_type = None
        # 表示边上数据在end节点执行函数中的参数位置，
        # 例如2^3，end节点为pow，则2对应的边的parameter_index=1，3对应的边的parameter_index=2
        # 例如3^2，end节点为pow，则3对应的边的parameter_index=1，2对应的边的parameter_index=2
        self.parameter_index = None
        self.condition = condition
        self.reverse = False
        if self.condition != 'no':
            self.need_var = kwargs['need_var']
            self.SplitCon()
        if in_var:
            self.var = in_var
        elif out_var:
            self.var = out_var
        self.start = start
        self.end = end

    def SplitCon(self):
        con_info = self.condition.split('$')
        if con_info[0] == 'T':
            self.reverse = True
            self.condition = con_info[1]
        else:
            self.condition = con_info[0]

    def GetStart(self):
        return self.start

    def GetEnd(self):
        return self.end

    def GetCondition(self):
        return self.reverse, self.condition


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    # 有关边的方法
    def InsertEdge(self, start, end, condition='no', **kwargs):
        edge = Edge(start, end, condition, **kwargs)
        self.edges.append(edge)
        start.out_edges.append(edge)
        end.in_edges.append(edge)

    def GetEdge(self, start, end):
        if len(self.edges):
            for edge in self.edges:
                if start == edge.GetStart() and end == edge.GetEnd():
                    return edge
            else:
                return False

    # 有关节点的方法
    def InsertNode(self, node):
        self.nodes.append(node)
        return True

    def GetNode(self, id):
        if len(self.nodes):
            for node in self.nodes:
                if id == node.GetId():
                    return node
            else:
                return False
        else:
            return False

    def GetSet(self):
        return self.nodes, self.edges

    # 其它方法
    def Show(self):
        dot = Digraph(name="computation graph", format="png")
        for node in self.nodes:
            id = node.GetId()
            dot.node(name=str(id), label=str(id) + '\n' + str(node.__class__))
        for edge in self.edges:
            if edge.GetCondition()[1] == 'no':
                dot.edge(str(edge.GetStart().GetId()), str(edge.GetEnd().GetId()),
                         label=edge.GetCondition()[1], color='green')
            elif edge.GetCondition()[0]:
                dot.edge(str(edge.GetStart().GetId()), str(edge.GetEnd().GetId()),
                         label='!' + edge.GetCondition()[1], color='red')
            else:
                dot.edge(str(edge.GetStart().GetId()), str(edge.GetEnd().GetId()),
                         label=edge.GetCondition()[1], color='yellow')
        dot.view(filename="my picture")

    def Merge(self, m_set):
        self.nodes = self.nodes + m_set[0]
        self.edges = self.edges + m_set[1]

    def ConvertToMatrix(self):
        matrix = np.zeros((3, len(self.nodes)+1, len(self.nodes)+1), dtype='np.float')
        for e in self.edges:
            matrix[0][e.start.GetId()][e.end.GetId()] = 1
            matrix[1][e.start.GetId()][e.end.GetId()] = e.condition
            matrix[2][e.start.GetId()][e.end.GetId()] = e.reverse
        return matrix


if __name__ == '__main__':
    G = Graph()
    root = nd.InstantiationClass(0, 'Root')
    X = nd.InstantiationClass(1, 'Root')
    Y = nd.InstantiationClass(2, 'Root')
    LR = nd.InstantiationClass(3, 'Root')
    Val = nd.InstantiationClass(4, 'Root')
    ass1 = nd.InstantiationClass(5, 'Root')
    W = nd.InstantiationClass(6, 'Root')
    Ran = nd.InstantiationClass(7, 'Root')
    ass2 = nd.InstantiationClass(8, 'Root')
    G.InsertNode(root)
    G.InsertNode(X)
    G.InsertNode(Y)
    G.InsertNode(Val)
    G.InsertNode(LR)
    G.InsertNode(W)
    G.InsertNode(Ran)
    G.InsertEdge(root, X)
    G.InsertEdge(root, Y)
    G.InsertEdge(root, LR)
    G.InsertEdge(X, W)
    G.InsertEdge(Y, W)
    G.InsertEdge(W, Val)
    G.InsertEdge(Val, Ran)
    print(G.ConvertToMatrix())
    G.Show()