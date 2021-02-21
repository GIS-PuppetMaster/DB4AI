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
            if self.IsGreateTensor(query):
                print(query)
            pass
        self.graph.Show()
    def ChangeRoot(self,id):
        self.root_id = id

    def IsGreateTensor(self,str):
        # 用于匹配的正则表达式
        variable_name_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*)'
        data_list_reg = '[(]([\d,]?(-1,)?)*([\d])[)]|[(]([\d,]?(-1,)?)*(-1)[)]|[(]-1,[)]|[(][\d],[)]'
        create_tensor_reg = f'^CREAT TENSOR {variable_name_reg}[^ ]*( FROM [^ ]+)?( WITH [^ ]+)?( AS [A-Z]+)?'
        val_info_reg1 = '[1-9]+[.][\d]+|0[.][\d]+|[1-9]+[\d]*|0'
        # val_info_reg2 = 'table'
        val_info_reg3 = 'RANDOM[(][(]-1,[\d][)][)]|RANDOM[(][(][\d],-1[)][)]|RANDOM[(][(][\d],[\d][)][)]'

        data_list_pattern = re.compile(data_list_reg)
        # 对读入的字符进行匹配检验是否合法和提取信息
        flag = True  # 用于记录匹配字符是否合法
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
                    flag = False
                    print('匹配失败2')
                    return flag
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
                        flag = False
                        print('匹配失败3')
                        return flag
                if len(with_str) != 0:
                    pass
                if len(as_str) != 0:
                    pass
        else:
            flag = False
            print('匹配失败1')
            return flag
        # 对上一步的结果进行处理
        self.node_id += 1
        if hasData:
            node1 = Nd.InstantiationClass(self.node_id, 'CreatTensor', data_shape=legal_info[1])
        else:
            node1 = Nd.InstantiationClass(self.node_id, 'CreatTensor', data_shape='()')
        self.graph.InsertNode(node1)
        self.graph.InsertEdge(self.root_id,self.node_id)
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
            self.graph.InsertEdge(self.node_id,node1_id)
        return flag

    def IsLoop(self):
        pass
    def IsIf(self):
        pass
if __name__ == '__main__':
    testList = []
    testList.append('CREAT TENSOR X(-1,4)')
    testList.append('CREAT TENSOR Y(-1,1)')
    testList.append('CREAT TENSOR LR(1,) FROM 5')
    testList.append('CREAT TENSOR W FROM RANDOM((1,4)) WITH GRAD')
    testPar = Parser(testList)
    testPar()

