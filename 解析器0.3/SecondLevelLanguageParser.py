import Digraph as DG
import Nodes as Nd
from json import loads
import re


class Parser:
    def __init__(self, queries: list):
        self.queries = queries
        self.var_dict = dict()
        self.exp_list = dict()
        self.graph = DG.Graph()
        self.node_id = 0
        self.root_id = 0
        self.in_loop = False
        self.loop_use = list()

    def __call__(self, *args, **kwargs):
        print(self.queries)
        if self.root_id == 0:
            root = Nd.InstantiationClass(self.node_id, 'Root')
            self.graph.InsertNode(root)
        for query in self.queries:
            if self.CreateTensor(query):
                # print('create合法语句:' + query)
                pass
            elif self.Loop(query):
                # print('loop合法语句:' + query)
                pass
            elif self.If(query):
                print('if合法语句:'+query)
                pass
        # self.graph.Show()

    def ChangeRoot(self, now_root):
        self.root_id = now_root

    def UpdateVarList(self, v_name, nd_id):
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

    def CreateTensor(self, query):
        # 用于匹配的正则表达式
        variable_name_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*)'
        data_list_reg = '[(]([1-9,]?(-1,)?)*([1-9])[)]|[(]([1-9,]?(-1,)?)*(-1)[)]|[(]-1,[)]|[(][1-9],[)]'
        create_tensor_reg = f'^CREATE TENSOR {variable_name_reg}[^ ]*( FROM [^ ]+)?( WITH [^ ]+)?( AS [A-Z]+)?'
        val_info_reg1 = '[1-9]+[.][0-9]+|0[.][0-9]+|[1-9]+[0-9]*|0'
        val_info_reg2 = variable_name_reg  # 暂时考虑使用变量名的要求
        val_info_reg3 = 'RANDOM[(][(]-1,[0-9][)][)]|RANDOM[(][(][0-9],-1[)][)]|RANDOM[(][(][0-9],[0-9][)][)]'

        data_list_pattern = re.compile(data_list_reg)
        # 对读入的字符进行匹配检验是否合法和提取信息
        hasData = False  # 所建的Tensor要存数据
        hasFrom = 0  # 赋值结点类型
        legal_info = []  # 记录合法的信息
        matchObj = re.match(create_tensor_reg, query)
        if matchObj:
            query = matchObj.group()
            T_name = matchObj.group(1)
            li = query.split()
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
            if len(li) > 3:
                from_str = ''
                with_str = ''
                as_str = ''
                for i in range(3, len(li) - 1):
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
                    if re.match(val_info_reg1, from_str):
                        value_str = re.match(val_info_reg1, from_str).group()
                        if re.search('[.]', value_str):
                            value = float(value_str)
                        else:
                            value = int(value_str)
                        legal_info.append(value)
                        hasFrom = 1
                    elif re.match(val_info_reg2, from_str):
                        t_name = re.match(val_info_reg2, from_str).group()
                        if len(t_name) > 30:
                            # print('匹配失败3')
                            return False
                        legal_info.append(t_name)
                        hasFrom = 2
                    elif re.match(val_info_reg3, from_str):
                        data_list_str = data_list_pattern.search(from_str).group()
                        data_list = loads(data_list_str.replace('(', '[').replace(')', ']'))
                        legal_info.append(data_list)
                        hasFrom = 3
                    else:
                        # print('匹配失败3')
                        return False
                if len(with_str) != 0:
                    # 待补充，关于 with语句的处理
                    pass
                if len(as_str) != 0:
                    # 待补充，关于as语句的处理
                    pass
        else:
            # print('匹配失败1')
            return False
        # 对上一步的结果进行处理
        self.node_id += 1
        if hasData:
            node1 = Nd.InstantiationClass(self.node_id, 'CreateTensor', data_shape=legal_info[1])
        else:
            node1 = Nd.InstantiationClass(self.node_id, 'CreateTensor', data_shape='()')
        self.graph.InsertNode(node1)
        self.graph.InsertEdge(self.root_id, self.node_id)
        if hasFrom != 0:
            if hasData:
                from_info = legal_info[2]
            else:
                from_info = legal_info[1]
            node1_id = self.node_id
            self.node_id += 1
            if hasFrom == 1:
                node2 = Nd.InstantiationClass(self.node_id, 'Val', value=from_info)
            elif hasFrom == 2:
                node2 = Nd.InstantiationClass(self.node_id, 'Sql', t_name=from_info)
            else:
                up = from_info[0]
                low = from_info[1]
                node2 = Nd.InstantiationClass(self.node_id, 'Random', uLimit=up, lLimit=low)
            self.graph.InsertNode(node2)
            node2_id = self.node_id
            self.node_id += 1
            node3 = Nd.InstantiationClass(self.node_id, 'Assignment')
            self.graph.InsertNode(node3)
            self.graph.InsertEdge(node1_id, self.node_id)
            self.graph.InsertEdge(node2_id, self.node_id)
            self.UpdateVarList(legal_info[0], self.node_id)
            if self.in_loop:
                self.loop_use.append(legal_info[0])  # 记录loop内使用的变量
        return True

    def Loop(self, query):
        loop_reg = 'LOOP ([1-9][0-9]*){\n((.*\n)(break\n)?)*}'
        matchObj = re.match(loop_reg, query)
        if matchObj:
            times = matchObj.group(1)
            in_str = re.search('{\n((.*\n)(break\n)?)*}', query).group()
            in_str_li = in_str.split('\n')
            in_str_li = in_str_li[1:len(in_str)-1]
            self.node_id += 1
            node1 = Nd.InstantiationClass(self.node_id, 'Loop', times=times)
            self.graph.InsertNode(node1)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n, self.node_id)
            self.ChangeRoot(self.node_id)
            node1_id = self.node_id
            queries = self.queries  # 调用call解析内部语句时，保存未处理完的字串
            self.queries = in_str_li
            self.in_loop = True
            self.__call__()
            self.node_id += 1
            node2 = Nd.InstantiationClass(self.node_id, 'Loop_End')
            self.graph.InsertNode(node2)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n, self.node_id)
            self.graph.InsertEdge(self.node_id, node1_id)
            for v in self.loop_use:
                last_use = self.var_dict.get(v)
                self.graph.InsertEdge(last_use[-1], node1_id)
                self.UpdateVarList(v, node1_id)
            self.queries = queries
            self.in_loop = False
            self.ChangeRoot(self.node_id)
            return True
        else:
            # print('匹配失败')
            return False

    def If(self, query):
        if_reg = 'IF [^ ]+{\n(.*\n)+}( ELIF [^ ]+{\n(.*\n)+})*( ELSE{\n(.*\n)+})?'
        matchObj = re.match(if_reg, query)
        if matchObj:
            if_str = matchObj.group()
            if_end = ''
            if_str_li = []
            if re.search('ELIF', if_str):
                if_str_li = if_str.split(' ELIF ')
                if_heads = if_str_li[0].split('IF ')
                if_str_li = if_str_li[1:len(if_str_li)]
                if_str_li.reverse()
                if_str_li.append(if_heads[1])
                if_str_li.reverse()
                if re.search('ELSE', if_str):
                    if_ends = if_str_li[-1].split(' ELSE')
                    if_str_li = if_str_li[:len(if_str_li) - 1]
                    if_str_li.append(if_ends[0])
                    if_end = if_ends[1]
            else:
                if_heads = if_str.split('IF ')
                if_str_li.append(if_heads[1])
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'If')
            self.graph.InsertNode(node)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n, self.node_id)
            if_node_id = self.node_id
            queries = self.queries
            for i_sen in if_str_li:
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'If_Branch')
                self.graph.InsertNode(node)
                i_sens = i_sen.split('{')
                self.graph.InsertEdge(if_node_id, self.node_id, i_sens[0])
                in_str_li = i_sens[1].split('\n')
                self.queries = in_str_li[1:len(in_str_li)-1]
                self.ChangeRoot(self.node_id)
                self.__call__()
            if len(if_end) != 0:
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'If_Branch')
                self.graph.InsertNode(node)
                self.graph.InsertEdge(if_node_id, self.node_id)
                in_str_li = if_end.split('\n')
                self.queries = in_str_li[1:len(in_str_li) - 1]
                self.ChangeRoot(self.node_id)
                self.__call__()
            self.node_id += 1
            node2 = Nd.InstantiationClass(self.node_id, 'If_End')
            self.graph.InsertNode(node2)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n, self.node_id)
            self.ChangeRoot(self.node_id)
            self.queries = queries
            return True
        else:
            return False


if __name__ == '__main__':
    testList = list()
    # testList.append('CREATE TENSOR X(-1,4)')
    # testList.append('CREATE TENSOR Y(-1,1)')
    testList.append('IF a<1{\nCREATE TENSOR X(-1,4)\n} ELIF c>0.04{\nCREATE TENSOR Y(-1,1)\n} ELSE{\nCREATE TENSOR LR(1,) FROM 0.05\n}')
    # testList.append('CREATE TENSOR LR(1,) FROM 0.05')
    # testList.append(
    #     'LOOP 100{\nX = SELECT F1,F2,F3,F4 FROM INPUT\nDEF G = GRADIENT(LOSS,W)\nW += LR * G\nIF LOSS(X)<0.01\n}')
    testList.append('CREATE TENSOR W FROM RANDOM((1,4)) WITH GRAD')
    testPar = Parser(testList)
    testPar()
