import Nodes as nd
import GraphVisualization as gv
from queue import Queue as qu
class Edge:
    def __init__(self,start,end,condition):
        self.data_shape = None
        self.data_type = None
        self.data_physic_type = None
        # 表示边上数据在end节点执行函数中的参数位置，
        # 例如2^3，end节点为pow，则2对应的边的parameter_index=1，3对应的边的parameter_index=2
        # 例如3^2，end节点为pow，则3对应的边的parameter_index=1，2对应的边的parameter_index=2
        self.parameter_index = None
        self.condition = condition
        self.start = start
        self.end = end

    def GetStart(self):
        return self.start

    def GetEnd(self):
        return self.end

    def Set(self,start,end):
        self.start = start
        self.end = end

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.connect_info = dict()
    # 有关边的方法
    def InsertEdge(self,start,end,condition = 'no'):
        # if(len(self.edges)):
        #     for edge in self.edges:
        #         if(start==edge.GetStart() and end==edge.GetEnd()):
        #             return False
        edge = Edge(start,end,condition)
        self.edges.append(edge)
        return True

    # def DeleEdge(self,start,end):
    #     if (len(self.edges)):
    #         for edge in self.edges:
    #             if (start == edge.GetStart() and end == edge.GetEnd()):
    #                self.edges.remove(edge)
    #                return True
    #         else:
    #             return False
    #     else:
    #         return False

    # def SetEdge(self,start,end):
    #     if (len(self.edges)):
    #         for edge in self.edges:
    #             if (start == edge.GetStart() and end == edge.GetEnd()):
    #                self.edges.remove(edge)
    #                break
    #         else:
    #             return False
    #         edge = Edge(start, end)
    #         self.edges.append(edge)
    #         return True
    #     else:
    #         return False

    def GetEdge(self,start,end):
        if (len(self.edges)):
            for edge in self.edges:
                if (start == edge.GetStart() and end == edge.GetEnd()):
                    return edge
            else:
                return False
    # 有关结点的方法
    def InsertNode(self,node):
        # id = node.GetId
        # if (len(self.nodes)):
        #     for n in self.nodes:
        #         if (id == n.GetId()):
        #             return False
        self.nodes.append(node)
        return True

    # def DeleNode(self,id):
    #     if(len(self.nodes)):
    #         for node in self.nodes:
    #             if(id == node.GetId()):
    #                 self.nodes.remove(node)
    #                 for edge in self.edges:
    #                     if(edge.GetStart() == id or edge.GetEnd == id):
    #                         self.edges.remove(edge)
    #                 return True
    #         else:
    #             return False
    #     else:
    #         return False

    def GetNode(self,id):
        if (len(self.nodes)):
            for node in self.nodes:
                if (id == node.GetId()):
                    return node
            else:
                return False
        else:
                return False
    # 其它方法
    def Show(self):
        edges = []
        nodes = []
        for node in self.nodes:
            id = node.GetId()
            type = node.GetType()
            print('id:%d,type:%d' %(id,type))
            nodes.append(id)
        for edge in self.edges:
            eStart = edge.GetStart()
            eEnd = edge.GetEnd()
            edges.append((eStart,eEnd))
        gv.Show_Graph(edges,nodes)

    def GetLeafNode(self,root_id):
        Q = qu(0)
        leafs = []
        Q.put(root_id)
        is_leaf = True
        while not Q.empty():
            root = Q.get()
            for e in self.edges:
                if root == e.GetStart():
                    Q.put(e.GetEnd())
                    is_leaf = False
            if is_leaf:
                leafs.append(root)
            is_leaf = True
        return leafs

if __name__ == '__main__':
    G = Graph()
    root = nd.InstantiationClass(0,'Root')
    X = nd.InstantiationClass(1,'Root')
    Y = nd.InstantiationClass(2, 'Root')
    LR = nd.InstantiationClass(3,'Root')
    Val = nd.InstantiationClass(4,'Root')
    ass1 = nd.InstantiationClass(5,'Root')
    W = nd.InstantiationClass(6, 'Root')
    Ran = nd.InstantiationClass(7,'Root')
    ass2 = nd.InstantiationClass(8,'Root')
    G.InsertNode(root)
    G.InsertNode(X)
    G.InsertNode(Y)
    G.InsertNode(Val)
    G.InsertNode(LR)
    G.InsertNode(ass1)
    G.InsertNode(ass2)
    G.InsertNode(W)
    G.InsertNode(Ran)
    G.InsertEdge(0,1)
    G.InsertEdge(0,2)
    G.InsertEdge(0,3)
    G.InsertEdge(0,6)
    G.InsertEdge(3,5)
    G.InsertEdge(4,5)
    G.InsertEdge(6,8)
    G.InsertEdge(7,8)
    li = G.GetLeafNode(0)
    print(li)
    G.Show()

