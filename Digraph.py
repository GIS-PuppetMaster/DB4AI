import Nodes as nd
import numpy as np
from graphviz import Digraph
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
        if condition == 'no':
            condition = None
        self.condition = condition
        self.reverse = False
        if self.condition is not None:
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
        self.without_out = set()
        self.exist_edge = dict(dict())

    # 有关边的方法
    def InsertEdge(self, start, end, condition='no', **kwargs):
        if start.id in self.exist_edge.keys() and end.id in self.exist_edge[start.id].keys():
            return False
        else:
            edge = Edge(start, end, condition, **kwargs)
            self.edges.append(edge)
            end_dict = {end.id: 1}
            self.exist_edge[start.id] = end_dict
            if start in self.without_out and not (isinstance(start, nd.LoopEnd) and isinstance(end, nd.Loop)):
                self.without_out.remove(start)
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
        self.without_out.add(node)
        return True

    def GetSet(self):
        return self.nodes, self.edges

    # 其它方法
    def Show(self):
        dot = Digraph(name="computation graph", format="png")
        for node in self.nodes:
            node_info = str(node.id) + '\n' + str(node.__class__) + '\n' + str(node.branches) + \
                        '\n' + str(node.vars) + '\n' + str(node.with_grad)
            dot.node(name=str(node.id), label=node_info)
        for edge in self.edges:
            if not edge.GetCondition()[1]:
                dot.edge(str(edge.GetStart().id), str(edge.GetEnd().id),
                         label=edge.GetCondition()[1], color='green')
            elif edge.GetCondition()[0]:
                dot.edge(str(edge.GetStart().id), str(edge.GetEnd().id),
                         label='!' + edge.GetCondition()[1], color='red')
            else:
                dot.edge(str(edge.GetStart().id), str(edge.GetEnd().id),
                         label=edge.GetCondition()[1], color='yellow')
        dot.view(filename="my picture")

    def Merge(self, m_set):
        self.nodes = self.nodes + m_set[0]
        self.edges = self.edges + m_set[1]

    def ConvertToMatrix(self):
        matrix = np.zeros((1, len(self.nodes) + 1, len(self.nodes) + 1))
        for e in self.edges:
            matrix[0][e.start.id][e.end.id] = 1
        return matrix

    def GetNoOutNodes(self):
        return self.without_out

    def ChangeNodeInfo(self, s_id, branches, with_grad):
        for i in range(len(self.nodes)):
            self.nodes[i].id += s_id
            del self.nodes[i].branches[0]
            for j in range(len(self.nodes[i].branches)):
                self.nodes[i].branches[j] += s_id
            self.nodes[i].branches = branches + self.nodes[i].branches
            self.nodes[i].with_grad = with_grad


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
    G.InsertEdge(Y, W)
    print(G.ConvertToMatrix())
    G.Show()
