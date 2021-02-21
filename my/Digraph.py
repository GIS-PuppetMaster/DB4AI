import Nodes as nd
import GraphVisualization as gv

class Edge:
    def __init__(self,start,end):
        self.data_shape = None
        self.data_type = None
        self.data_physic_type = None
        # 表示边上数据在end节点执行函数中的参数位置，
        # 例如2^3，end节点为pow，则2对应的边的parameter_index=1，3对应的边的parameter_index=2
        # 例如3^2，end节点为pow，则3对应的边的parameter_index=1，2对应的边的parameter_index=2
        self.parameter_index = None
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
    # 有关边的方法
    def InsertEdge(self,start,end):
        if(len(self.edges)):
            for edge in self.edges:
                if(start==edge.GetStart() and end==edge.GetEnd()):
                    return False
        edge = Edge(start,end)
        self.edges.append(edge)
        return True

    def DeleEdge(self,start,end):
        if (len(self.edges)):
            for edge in self.edges:
                if (start == edge.GetStart() and end == edge.GetEnd()):
                   self.edges.remove(edge)
                   return True
            else:
                return False
        else:
            return False

    def SetEdge(self,start,end):
        if (len(self.edges)):
            for edge in self.edges:
                if (start == edge.GetStart() and end == edge.GetEnd()):
                   self.edges.remove(edge)
                   break
            else:
                return False
            edge = Edge(start, end)
            self.edges.append(edge)
            return True
        else:
            return False
    # 有关结点的方法
    def InsertNode(self,node):
        id = node.GetId
        if (len(self.nodes)):
            for n in self.nodes:
                if (id == n.GetId()):
                    return False
        self.nodes.append(node)
        return True

    def DeleNode(self,id):
        if(len(self.nodes)):
            for node in self.nodes:
                if(id == node.GetId()):
                    self.nodes.remove(node)
                    for edge in self.edges:
                        if(edge.GetStart() == id or edge.GetEnd == id):
                            self.edges.remove(edge)
                    return True
            else:
                return False
        else:
            return False

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

if __name__ == '__main__':
    G = Graph()
    root = nd.InstantiationClass(0,'Root')
    G.InsertNode(root)
    tensorX = nd.InstantiationClass(1,'CreatTensor',data_shape='(2,3,4)')
    tensorY = nd.InstantiationClass(2, 'CreatTensor', data_shape='(1,2,3)')
    Val = nd.InstantiationClass(3,'Val',value=12)
    G.InsertNode(tensorX)
    G.InsertNode(tensorY)
    G.InsertNode(Val)
    G.InsertEdge(0,1)
    G.InsertEdge(0,2)
    G.InsertEdge(3,2)
    tensorZ = nd.InstantiationClass(4,'CreatTensor',data_shape='(3,6,8)')
    G.InsertNode(tensorZ)
    G.Show()

