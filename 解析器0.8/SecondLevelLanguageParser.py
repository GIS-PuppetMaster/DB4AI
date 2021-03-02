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
        self.state_stack = list()
        self.loop_or_if_id = 0
        self.out_var = list()
        self.in_var = list()
        self.oth_branch = 0

    def __call__(self, *args, **kwargs):
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
            elif self.End(query):
                pass
            else:
                print('非法语句：'+query)
        self.graph.Show()

    #  用于解析语句时维护解析器或计算图数据的主要函数
    def StateConvert(self, c_state):
        """
        为了解析loop和if语句，把解析器抽象成一个自动机，通过当前状态+读取的状态来切换状态并修改相关数值
        :param c_state: 要求切换到的状态
        :return: 无
        """
        if len(self.state) == 0 and (c_state == 'loop' or c_state == 'if'):
            self.root_id = self.node_id
            self.state = c_state
            self.out_var = self.var_dict.keys()
            self.loop_or_if_id = self.root_id
        elif (self.state == 'loop' or self.state == 'if_branch') and (c_state == 'loop' or c_state == 'if'):
            if self.state == 'if_branch' and c_state == 'if':
                self.state_stack.append([self.loop_or_if_id, self.state, self.out_var, self.in_var, self.oth_branch])
            else:
                self.state_stack.append([self.loop_or_if_id, self.state, self.out_var, self.in_var])
            self.root_id = self.node_id
            self.state = c_state
            self.out_var = self.var_dict.keys()
            self.loop_or_if_id = self.root_id
        elif self.state == 'if' and c_state == 'if_branch':
            self.root_id = self.node_id
            self.loop_or_if_id = self.root_id
            self.state = c_state
            self.oth_branch = self.node_id
        elif self.state == 'if_branch' and c_state == 'end':
            self.state = 'if'
        elif (self.state == 'if' or self.state == 'loop') and c_state == 'end':
            self.root_id = self.node_id
            if len(self.state_stack) == 0:
                self.state = ''
            else:
                state_li = self.state_stack[-1]
                if state_li[1] == 'if_branch' and self.state == 'if':
                    self.oth_branch = state_li[4]
                self.in_var = state_li[3]
                self.out_var = state_li[2]
                self.state = state_li[1]
                self.loop_or_if_id = state_li[0]

    def UpdateVarList(self, v_name, nd_id):
        """
        用于维护变量名列表的函数
        :param v_name: 需要维护的变量名
        :param nd_id: 该变量名对应的最近一次赋值的节点
        :return: 无
        """
        var_li = self.var_dict.get(v_name, None)
        if var_li:
            new_li = var_li
            new_li.append(nd_id)
            self.var_dict[v_name] = new_li
        else:
            new_li = list()
            new_li.append(nd_id)
            self.var_dict[v_name] = new_li
            
    def UpdateExpList(self, n_exp, v_node, v_name):
        """
        用于维护表达式列表的函数
        :param n_exp: 需要维护的表达式
        :param v_node: 表达式对应的子图与其他部分成“桥”的节点
        :param v_name: 表达式所赋值的变量
        :return: 无
        """
        exp, node = self.exp_list.get(v_name, None)
        if exp:
            self.exp_list[v_name] = (n_exp, v_node)
        else:
            if exp != n_exp:
                self.exp_list[v_name] = (n_exp, v_node)
        pass

    def EndIf(self):
        """
        用于结束if语句
        :return: 无
        """
        if self.state == 'if':
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'If_End')
            self.graph.InsertNode(node)
            leaf_li = self.graph.GetLeafNode(self.root_id)
            for l_n in leaf_li:
                self.graph.InsertEdge(l_n, self.node_id)
            self.ConnInVar(self.node_id)
            self.StateConvert('end')  # 以if状态下非if型语句解析结束if状态

    def DealInVar(self, v_name, is_new=False):
        '''
        处理循环结构和条件结构内部的变量，连接内部用到的外部变量和"收集"内部新建变量
        :param v_name: 变量名
        :param is_new: 是否为新建变量
        :return: 无
        '''
        if self.state == 'loop' or self.state == 'if_branch':
            if is_new:
                self.in_var.append(v_name)
            elif v_name in self.out_var:
                var_li = self.var_dict.get(v_name)
                last_use = var_li[-1]
                self.graph.InsertEdge(last_use, self.loop_or_if_id)
                self.graph.InsertEdge(self.loop_or_if_id, self.node_id)
                self.UpdateVarList(v_name, self.loop_or_if_id)

    def ConnInVar(self, e_node):
        '''
        连接循环结构和条件结构的内部新建变量到end节点
        :param e_node: end节点
        :return: 无
        '''
        for i_n in self.in_var:
            self.UpdateVarList(i_n, e_node)

    def Expression(self, query):
        # 待添加
        pass

    #  用于解析语句的主要函数
    def CreateTensor(self, query):
        """
        解析create语句的函数，对于合法的语句进行建图操作
        :param query: 需要解析的语句
        :return: True 语句合法，False 语句非法
        """
        self.EndIf()
        # 用于匹配的正则表达式
        variable_name_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*)'
        data_list_reg = '[(]([1-9,]?(-1,)?)*([1-9])[)]|[(]([1-9,]?(-1,)?)*(-1)[)]|[(]-1,[)]|[(][1-9],[)]'
        create_tensor_reg = f'^CREATE TENSOR {variable_name_reg}[^ ]*( FROM [^ ]+)?( WITH GRAD)?( AS [A-Z]+)?$'
        val_info_reg1 = '[1-9]+[.][0-9]+|0[.][0-9]+|[1-9]+[0-9]*|0'
        val_info_reg2 = variable_name_reg  # 暂时考虑使用变量名的要求,待修改
        val_info_reg3 = 'RANDOM[(][(]-1,[0-9][)][)]|RANDOM[(][(][0-9],-1[)][)]|RANDOM[(][(][0-9],[0-9][)][)]'

        data_list_pattern = re.compile(data_list_reg)
        # 对读入的字符进行匹配检验是否合法和提取信息
        hasData = False  # 所建的Tensor要存数据
        hasWith = False  # 是否需要记录梯度
        hasFrom = 0  # 赋值结点类型
        legal_info = []  # 记录合法的信息
        matchObj = re.match(create_tensor_reg, query)
        if matchObj:
            query = matchObj.group()
            if re.search('WITH GRAD', query):
                hasWith = True
            T_name = matchObj.group(1)
            li = query.split(' ')
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
                as_str = ''
                for i in range(3, len(li) - 1):
                    if re.search('FROM', li[i]):
                        j = i + 1
                        from_str = li[j]
                    elif re.search('AS', li[i]):
                        j = i + 1
                        as_str = li[j]
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
                node2 = Nd.InstantiationClass(self.node_id, 'Val')
                node2.set_val(legal_info[0])
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
            self.DealInVar(legal_info[0], True)
        self.UpdateVarList(legal_info[0], self.node_id)
        return True

    def Define(self, query):
        """
        解析define语句的函数，对于合法的语句进行建图操作
        :param query: 需要解析的语句
        :return: True 语句合法，False 语句非法
        """
        self.EndIf()
        variable_name_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*)'
        define_reg = f'^DEF {variable_name_reg} = (.+)\n$'
        matchObj = re.match(define_reg, query)
        if matchObj:
            v_name = matchObj.group(1)
            ass_exp = matchObj.group(2)
            node = self.Expression(ass_exp)
            if node:
                self.UpdateExpList(ass_exp, node, v_name)

    def Loop(self, query):
        """
        解析loop语句的函数，对于合法的语句进行建图
        :param query: 需要解析的语句
        :return:True 合法语句，False 非法语句
        """
        self.EndIf()
        loop_reg = 'LOOP ([1-9][0-9]*){\n'
        matchObj = re.match(loop_reg, query)
        if matchObj:
            loop_str = matchObj.group(1)
            times = int(re.search('[1-9][0-9]*', loop_str).group())
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
        """
        解析if语句的函数，对于合法的语句进行建图
        :param query: 需要解析的语句
        :return:True 合法语句，False 非法语句
        """
        if_reg = 'IF [^ ]+{\n'  # 所有关于if的正则，目前未对条件进行约束，待修改
        elif_reg = 'ELIF [^ ]+{\n'
        else_reg = 'ELSE{\n'
        matchObj_if = re.match(if_reg, query)
        matchObj_elif = re.match(elif_reg, query)
        matchObj_else = re.match(else_reg, query)
        if matchObj_if:
            # TODO: 无法正常生成IF_END节点，请检查状态机
            self.EndIf()
            if_str = matchObj_if.group(0)
            condition = re.search('[^ ]', if_str).group()
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
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'If_Branch')
            self.graph.InsertNode(node)
            self.graph.InsertEdge(self.root_id, self.node_id, '!'+condition)
            self.StateConvert('if_branch')
            return True
        elif matchObj_elif:
            if self.state == 'if':
                if_str = matchObj_elif.group(0)
                condition = re.search('[^ ]', if_str).group()
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'If_Branch')
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.oth_branch, self.node_id, condition)
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'If_Branch')
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.oth_branch, self.node_id, '!' + condition)
                self.StateConvert('if_branch')
            return True
        elif matchObj_else:
            if self.state == 'if':
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'If_Branch')
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.oth_branch, self.node_id)
                self.StateConvert('if_branch')
            return True
        else:
            return False

    def End(self, query):
        """
        识别\n}，结束当前语句结构
        :param query: 需要解析的语句
        :return:True 合法语句，False 非法语句
        """
        end_reg = '\n}'
        matchObj = re.match(end_reg, query)
        if matchObj:
            if self.state == 'loop':
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'Loop_End')
                self.graph.InsertNode(node)
                leaf_li = self.graph.GetLeafNode(self.root_id)
                for l_n in leaf_li:
                    self.graph.InsertEdge(l_n, self.node_id)
                self.ConnInVar(self.node_id)
            self.StateConvert('end')
            return True
        else:
            return False

    def Assignment(self, query):
        """
        解析assignment语句的函数，对于合法的语句进行建图操作
        :param query: 需要解析的语句
        :return: True 语句合法，False 语句非法
        """
        self.EndIf()
        variable_name_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*)'
        ass_reg = f'^{variable_name_reg} = (.+)\n'
        sql_reg = 'SELECT [a-zA-Z_]+[a-zA-Z0-9_]* FROM ([a-zA-Z_]+[a-zA-Z0-9_]*)( WHERE [.]+)?'
        matchObj = re.match(ass_reg, query)
        if matchObj:
            v_name = matchObj.group(1)
            ass_exp = matchObj.group(2)
            exp, node = self.exp_list.get(v_name, None)
            if node:
                if ass_exp == exp:
                    e_node = node
                    e_node_id = e_node.GetId()
                else:
                    return False
            else:
                ass_str = matchObj.group(2)
                print(ass_str)
                sql_matchObj = re.match(sql_reg, ass_str)
                if sql_matchObj:
                    t_name = sql_matchObj.group(1)
                    self.node_id += 1
                    e_node = Nd.InstantiationClass(self.node_id, 'Sql', t_name=t_name)
                    e_node_id = self.node_id
                    self.graph.InsertNode(e_node)
                else:
                    return False
        else:
            return False
        var_li = self.var_dict.get(v_name, None)
        if var_li:
            self.DealInVar(v_name, False)
            last_use = var_li[-1]
            self.node_id += 1
            ass_n = Nd.InstantiationClass(self.node_id, 'Assignment')
            self.graph.InsertNode(ass_n)
            self.graph.InsertEdge(last_use, self.node_id)
        else:
            self.DealInVar(v_name, True)
            self.node_id += 1
            node_l = Nd.InstantiationClass(self.node_id, 'CreateTensor', data_shape='()')
            self.graph.InsertNode(node_l)
            node_l_id = self.node_id
            self.node_id += 1
            ass_n = Nd.InstantiationClass(self.node_id, 'Assignment')
            self.graph.InsertNode(ass_n)
            self.graph.InsertEdge(node_l_id, self.node_id)
        self.graph.InsertEdge(e_node_id, self.node_id)
        self.UpdateVarList(v_name, self.node_id)
        return True


if __name__ == '__main__':
    testList = list()
    testList.append('CREATE TENSOR X(-1,4)')
    testList.append('CREATE TENSOR Y(-1,1)')
    testList.append('IF a<1{\n CREATE TENSOR X(-1,4)\n}ELIF c>0.04{\nCREATE TENSOR Y(-1,1)\n}ELSE{\nCREATE TENSOR LR(1,) FROM 0.05\n}')
    testList.append('CREATE TENSOR LR(1,) FROM 0.05')
    # testList.append('LOOP 100{\n
    # X = SELECT F1,F2,F3,F4 FROM INPUT\n
    # DEF G = GRADIENT(LOSS,W)\n
    # W += LR * G\n
    # IF LOSS(X)<0.01
    # \n}')
    # testList.append('CREATE TENSOR W FROM RANDOM((1,4)) WITH GRAD')
    testPar = Parser(testList)
    testPar()
