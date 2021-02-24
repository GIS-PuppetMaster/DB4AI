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
        # 记录”状态“，用于loop和if语句的解析使用
        self.root_id = 0
        self.state = ''
        self.in_var = list()
        self.state_stack = list()
        self.loop_id = 0
        self.out_loop = list()

    def __call__(self, *args, **kwargs):
        print(self.queries)
        if self.root_id == 0:
            root = Nd.InstantiationClass(self.node_id, 'Root')
            self.graph.InsertNode(root)
        for query in self.queries:
            if self.CreateTensor(query):
                pass
            elif self.Loop(query):
                pass
            elif self.If(query):
                pass
            elif self.IfOrLoopEnd(query):
                pass
            else:
                print('非法语句：'+query)
        # self.graph.Show()

    def StateConvert(self, c_state):
        if len(self.state) == 0 and (c_state == 'loop' or c_state == 'if'):
            self.root_id = self.node_id
            self.state = c_state
            if c_state == 'loop':
                self.loop_id = self.root_id
        elif self.state == 'loop' and (c_state == 'loop' or c_state == 'if'):
            self.state_stack.append([self.loop_id, self.state, self.in_var])
            self.root_id = self.node_id
            self.state = c_state
            if c_state == 'loop':
                self.loop_id = self.root_id
        elif self.state == 'if' and c_state == 'if_branch':
            self.state_stack.append([self.root_id, self.state])
            self.root_id = self.node_id
            self.state = c_state
        elif self.state == 'if_branch' and (c_state == 'loop' or c_state == 'if'):
            self.state_stack.append([self.root_id,self.state])
            if c_state == 'loop':
                self.loop_id = self.root_id
        elif self.state == 'if_branch' and c_state == 'end':
            state_li = self.state_stack[-1]
            self.root_id = state_li[0]
            self.state = state_li[1]
        elif (self.state == 'if' or self.state == 'loop') and c_state == 'end':
            self.root_id = self.node_id
            if len(self.state_stack) == 0:
                self.state = ''
                self.in_var.clear()
            else:
                state_li = self.state_stack[-1]
                self.state = state_li[1]
                if self.state == 'loop':
                    self.in_var = state_li[2] + self.in_var
                    self.loop_id = state_li[0]

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
        if self.state == 'if':
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id,'If_End')
            self.graph.InsertNode(node)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n, self.node_id)
            self.StateConvert('end') # 以if状态下非if型语句解析结束if状态
        # 用于匹配的正则表达式
        variable_name_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*)'
        data_list_reg = '[(]([1-9,]?(-1,)?)*([1-9])[)]|[(]([1-9,]?(-1,)?)*(-1)[)]|[(]-1,[)]|[(][1-9],[)]'
        create_tensor_reg = f'^CREATE TENSOR {variable_name_reg}[^ ]*( FROM [^ ]+)?( WITH [^ ]+)?( AS [A-Z]+)?'
        val_info_reg1 = '[1-9]+[.][0-9]+|0[.][0-9]+|[1-9]+[0-9]*|0'
        val_info_reg2 = variable_name_reg  # 暂时考虑使用变量名的要求,待修改
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
                            return False
                        legal_info.append(t_name)
                        hasFrom = 2
                    elif re.match(val_info_reg3, from_str):
                        data_list_str = data_list_pattern.search(from_str).group()
                        data_list = loads(data_list_str.replace('(', '[').replace(')', ']'))
                        legal_info.append(data_list)
                        hasFrom = 3
                    else:
                        return False
                if len(with_str) != 0:
                    # 待补充，关于 with语句的处理
                    pass
                if len(as_str) != 0:
                    # 待补充，关于as语句的处理
                    pass
        else:
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
        else:
            self.UpdateVarList(legal_info[0],self.node_id)
        return True

    def Loop(self, query):
        loop_reg = 'LOOP ([1-9][0-9]*){\n'
        matchObj = re.match(loop_reg, query)
        if matchObj:
            loop_str = matchObj.group(1)
            times = int(re.search('[1-9][0-9]*',loop_str).group())
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'Loop', times=times)
            self.graph.InsertNode(node)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n, self.node_id)
            self.StateConvert('loop')
            return True
        else:
            return False

    def If(self, query):
        if_reg = 'IF [^ ]+{\n' # 所有关于if的正则，目前未对条件进行约束，待修改
        elif_reg = 'ELIF [^ ]+{\n'
        else_reg = 'ELSE{\n'
        matchObj_if = re.match(if_reg, query)
        matchObj_elif = re.match(elif_reg, query)
        matchObj_else = re.match(else_reg, query)
        if matchObj_if:
            if_str = matchObj_if.group(1)
            condition = re.search('[^ ]', if_str)
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'If')
            self.graph.InsertNode(node)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n, self.node_id)
            self.StateConvert('if')
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'If_Branch')
            self.graph.InsertNode(node)
            self.graph.InsertEdge(self.root_id, self.node_id, condition)
            self.StateConvert('if_branch')
            return True
        elif matchObj_elif:
            if self.state == 'if':
                if_str = matchObj_elif.group(1)
                condition = re.search('[^ ]', if_str)
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'If_Branch')
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.root_id, self.node_id, condition)
                self.StateConvert('if_branch')
            return True
        elif matchObj_else:
            if self.state == 'if':
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'If_Branch')
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.root_id, self.node_id)
                self.StateConvert('if_branch')
            return True
        else:
            return False

    def IfOrLoopEnd(self,query):
        end_reg = '\n}'
        matchObj = re.match(end_reg,query)
        if matchObj:
            self.StateConvert('end')
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
