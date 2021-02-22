import Digraph as DG
import Nodes as Nd
from json import loads as strTolist
import re

class Parser:
    def __init__(self, queries:list):
        self.queries = queries
        self.var_dict = dict()
        self.exp_list = dict()
        self.graph = DG.Graph()
        self.node_id = 0
        self.root_id = 0

    def __call__(self, *args, **kwargs):
        root = Nd.InstantiationClass(self.node_id,'Root')
        self.graph.InsertNode(root)
        for query in self.queries:
            if self.GreateTensor(query):
                print('合法语句：'+query)
            elif self.Loop(query):
                print('合法语句：'+query)
        self.graph.Show()
    def ChangeRoot(self,id):
        self.root_id = id
    def UpdateList(self,v_name,nd_id):
        v_name = v_name
        nd_id = nd_id
        new_li = []
        if self.var_dict.get(v_name, False):
            new_li = self.var_dict.get(v_name)
            new_li.append(nd_id)
            self.var_dict[v_name] = new_li
        else:
            new_li.append(nd_id)
            self.var_dict[v_name] = new_li

    def GreateTensor(self,str):
        # 用于匹配的正则表达式
        variable_name_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*)'
        data_list_reg = '[(]([1-9,]?(-1,)?)*([1-9])[)]|[(]([1-9,]?(-1,)?)*(-1)[)]|[(]-1,[)]|[(][1-9],[)]'
        create_tensor_reg = f'^CREAT TENSOR {variable_name_reg}[^ ]*( FROM [^ ]+)?( WITH [^ ]+)?( AS [A-Z]+)?'
        val_info_reg1 = '[1-9]+[.][\d]+|0[.][\d]+|[1-9]+[\d]*|0'
        # val_info_reg2 = 'table'
        val_info_reg3 = 'RANDOM[(][(]-1,[\d][)][)]|RANDOM[(][(][\d],-1[)][)]|RANDOM[(][(][\d],[\d][)][)]'

        data_list_pattern = re.compile(data_list_reg)
        # 对读入的字符进行匹配检验是否合法和提取信息
        hasData = False # 所建的Tensor要存数据
        hasFrom = 0 # 赋值结点类型
        legal_info = []  # 记录合法的信息
        matchObj = re.match(create_tensor_reg,str)
        if matchObj:
            str = matchObj.group()
            T_name = matchObj.group(1)
            li = str.split()
            if len(T_name) != len(li[2]):
                data = data_list_pattern.search(li[2])
                if data:
                    data_shape = data.group()
                    legal_info.append(T_name)
                    legal_info.append(data_shape)
                    hasData = True
                else:
                    # print('匹配失败2')
                    return False
            else:
                legal_info.append(T_name)
            if len(li)>3:
                from_str = ''
                with_str = ''
                as_str = ''
                for i in range(3, len(li)-1):
                    if re.search('FROM', li[i]):
                        j = i + 1
                        from_str = li[j]
                    elif re.search('WITH', li[i]):
                        j = i + 1
                        with_str = li[j]
                    elif re.search('AS', li[i]):
                        j = i + 1
                        with_str = li[j]
                if len(from_str) != 0:
                    if re.match(val_info_reg1,from_str):
                        value_str = re.match(val_info_reg1,from_str).group()
                        if re.search('[.]',value_str):
                            value = float(value_str)
                        else:
                            value = int(value_str)
                        legal_info.append(value)
                        hasFrom = 1
                    elif re.match(val_info_reg3,from_str):
                        data_list_str = data_list_pattern.search(from_str).group()
                        data_list = strTolist(data_list_str.replace('(', '[').replace(')', ']'))
                        legal_info.append(data_list)
                        hasFrom = 3
                    else:
                        # print('匹配失败3')
                        return False
                if len(with_str) != 0:
                    pass
                if len(as_str) != 0:
                    pass
        else:
            # print('匹配失败1')
            return False
        # 对上一步的结果进行处理
        self.node_id += 1
        if hasData:
            node1 = Nd.InstantiationClass(self.node_id, 'CreatTensor', data_shape=legal_info[1])
        else:
            node1 = Nd.InstantiationClass(self.node_id, 'CreatTensor', data_shape='()')
        self.graph.InsertNode(node1)
        self.graph.InsertEdge(self.root_id,self.node_id)
        self.UpdateList(legal_info[0],self.node_id)
        if hasFrom != 0:
            if hasData:
                from_info = legal_info[2]
            else:
                from_info = legal_info[1]
            node1_id = self.node_id
            self.node_id += 1
            if hasFrom == 1:
                node2 = Nd.InstantiationClass(self.node_id,'Val',value=from_info)
            elif hasFrom == 2:
                node2 = None
                pass
            else:
                up = from_info[0]
                low = from_info[1]
                node2 = Nd.InstantiationClass(self.node_id,'Random',uLimit=up,lLimit=low)
            self.graph.InsertNode(node2)
            node2_id = self.node_id
            self.node_id += 1
            node3 = Nd.InstantiationClass(self.node_id,'Assignment')
            self.graph.InsertNode(node3)
            self.graph.InsertEdge(node1_id,self.node_id)
            self.graph.InsertEdge(node2_id,self.node_id)
        return True

    def Loop(self,str):
        loop_reg = 'LOOP ([1-9][0-9]*){\n((.*\n)(break\n)?)*}'
        matchObj = re.match(loop_reg,str)
        if matchObj:
            times = matchObj.group(1)
            in_str = re.search('{\n((.*\n)(break\n)?)*}',str).group()
            in_str = in_str.replace('{','\n').replace('}','\n')
            in_strs = in_str.split('\n')
            self.node_id += 1
            node1 = Nd.InstantiationClass(self.node_id,'Loop',times=times)
            self.graph.InsertNode(node1)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n,self.node_id)
            self.ChangeRoot(self.node_id)
            print('loop内语句：')
            print(in_strs) # 该处要修改为解析in_strs中的所有字符
            self.node_id += 1
            node2 = Nd.InstantiationClass(self.node_id, 'Loop_End')
            self.graph.InsertNode(node2)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n, self.node_id)
            return True
        else:
            # print('匹配失败')
            return False

    def If(self):
        pass
if __name__ == '__main__':
    testList = list()
    testList.append('CREAT TENSOR X(-1,4)')
    testList.append('CREAT TENSOR Y(-1,1)')
    testList.append('CREAT TENSOR LR(1,) FROM 5')
    testList.append('CREAT TENSOR W FROM RANDOM((1,4)) WITH GRAD')
    testList.append('LOOP 100{\nX = SELECT F1,F2,F3,F4 FROM INPUT\nDEF G = GRADIENT(LOSS,W)\nW += LR * G\nIF LOSS(X)<0.01\n}')
    print(type(testList))
    testPar = Parser(testList)
    testPar()


