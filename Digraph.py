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
        self.without_in = set()
        self.exist_edge = dict(dict())

    # 有关边的方法
    def InsertEdge(self, start, end, condition='no', **kwargs):
        if start.id in self.exist_edge.keys() and end.id in self.exist_edge[start.id].keys():
            return False
        else:
            edge = Edge(start, end, condition, **kwargs)
            self.edges.append(edge)
            if start.id in self.exist_edge.keys():
                pre_dict = self.exist_edge[start.id]
                pre_dict[end.id] = 1
            else:
                end_dict = {end.id: 1}
                self.exist_edge[start.id] = end_dict
            if start in self.without_out and not (isinstance(start, nd.LoopEnd) and isinstance(end, nd.Loop)):
                self.without_out.remove(start)
            if end in self.without_in:
                self.without_in.remove(end)
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
        self.without_in.add(node)
        return True

    def GetSet(self):
        return self.nodes.copy(), self.edges.copy(), self.without_in.copy(), self.without_out.copy()

    # 其它方法
    def Show(self):
        dot = Digraph(name="computation graph", format="svg")
        for node in self.nodes:
            node_info = 'id:' + str(node.id) + '\n' + 'type:' + str(node.__class__) + '\n' + 'area:' +str(node.branches)\
                        + '\n' + 'vars:' + str(node.vars) + '\n' + 'grad:' + str(node.with_grad) + '\n' + f'branches: {node.branches}\n'
            if isinstance(node, nd.Loop) or isinstance(node, nd.LoopEnd) or isinstance(node, nd.Break):
                node_info = node_info + 'loop_pair:' + str(node.loop_pair.id)
            elif isinstance(node, nd.IfBranch):
                if node.end_if_pair is not None:
                    node_info = node_info + 'end_if_pair:' + str(node.end_if_pair.id)
            elif isinstance(node, nd.Zeros) or isinstance(node, nd.Ones) or isinstance(node, nd.Full):
                node_info = node_info + 'data_shape:' + str(node.data_shape)
            elif isinstance(node, nd.Random):
                node_info = node_info + 'data_shape:' + str(node.data_shape) + '\n' + 'boundary:' + str(node.boundary)
            elif isinstance(node, nd.Sql):
                node_info = node_info + 'sql:' + node.t_search_sentences
                '''elif isinstance(node, nd.Val):
                node_info = node_info + 'value:' + node.value'''
            elif isinstance(node, nd.Assignment):
                node_info = node_info + 'slice:' + str(node._slice) + '\n' +'update:' + str(node.update)
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
            for v in range(len(self.nodes[i].vars)):
                if isinstance(self.nodes[i], nd.SaveTable) and self.nodes[i].vars[0] is None:
                    continue
                if self.nodes[i].vars[v].startswith('_') and self.nodes[i].vars[v][1:].isdigit():
                    self.nodes[i].vars[v] = '_' + str(eval(self.nodes[i].vars[v][1:]) + s_id)
                if self.nodes[i].vars[v].startswith('__') and self.nodes[i].vars[v][2:].isdigit():
                    self.nodes[i].vars[v] = '__' + str(eval(self.nodes[i].vars[v][2:]) + s_id)
            del self.nodes[i].branches[0]
            for j in range(len(self.nodes[i].branches)):
                self.nodes[i].branches[j] += s_id
            self.nodes[i].branches = branches + self.nodes[i].branches
            if with_grad:
                self.nodes[i].with_grad = with_grad

    def get_state(self):
        nodes_num = len(self.nodes)
        nodes_feature_num = 100
        # 0-无边，1-数据流边，2-控制流+数据流边
        adj_matrix = np.zeros(shape=(nodes_num, nodes_num))
        nodes_feature_matrix = np.zeros(shape=(nodes_num, nodes_feature_num))
        # 填充邻接矩阵
        for node_id, node in enumerate(self.nodes):
            assert node.id == node_id
            for in_edge in node.in_edges:
                start_node = in_edge.start
                start_node_id = in_edge.start.id
                if isinstance(start_node, nd.Loop) or isinstance(start_node, nd.If) or isinstance(start_node, nd.IfBranch) or isinstance(node, nd.LoopEnd) or isinstance(node, nd.IfEnd):
                    adj_matrix[start_node_id, node_id] = 2
                else:
                    adj_matrix[start_node_id, node_id] = 1
            for out_edge in node.in_edges:
                end_node = out_edge.end
                end_node_id = out_edge.end.id
                if isinstance(node, nd.Loop) or isinstance(node, nd.If) or isinstance(node, nd.IfBranch) or isinstance(end_node, nd.LoopEnd) or isinstance(end_node, nd.IfEnd):
                    adj_matrix[node_id, end_node_id] = 2
                else:
                    adj_matrix[node_id, end_node_id] = 1
        # 填充特征矩阵
        # for node_id, node in enumerate(self.nodes):
        #     if isinstance()
        #     nodes_feature_matrix[node_id, ...] = []
        return adj_matrix, nodes_feature_matrix


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
    # G.Show()
