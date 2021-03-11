import Digraph as DG
import Nodes as Nd
import re
import Analyze_expression as A_e
import random


class Parser:
    def __init__(self, queries: list):
        self.queries = queries
        self.var_dict = dict()
        self.graph = DG.Graph()
        self.node_id = 0
        self.current_node = None
        # 记录”状态“，用于特殊语句的解析使用
        self.root_id = 0
        self.current_root = None
        self.state = ''
        self.state_stack = list()
        self.loop_or_if_id = 0
        self.loop_id = 0
        self.out_var = list()
        self.in_var = list()
        self.oth_branch = 0
        self.leaf_li = set()
        #  用于自定义算子使用的特殊域
        self.input = list()
        self.output = list()
        self.pre_li = ['@', '#', '!', '$', '%', '^', '&', '*', '?', '+']  # 允许自定义100个算子
        self.pre = ''

    def __call__(self, **kwargs):
        """
        类的call方法，使用类中的语句解析方法解析语句列表，进行建图
        :param kwargs: start_id 整数 用于处理自定义算子情况的初始序号
        """
        is_cu = False
        in_queries = list()
        if len(kwargs) == 0:
            root = Nd.InstantiationClass(self.node_id, 'Root')
            self.graph.InsertNode(root)
            self.leaf_li.add(root)
        else:
            self.node_id = kwargs['start_id']
            self.pre = self.pre_li[random.randint(0, 9)]+self.pre_li[random.randint(0, 9)]
        for query in self.queries:
            if is_cu:
                if self.CuEnd(query):
                    in_queries.append(query)
                else:
                    in_parser = Parser(in_queries)
                    in_parser(start_id=self.node_id)
                    g_info = in_parser.GetGInfo()
                    # 待完善
            else:
                if self.CreateTensor(query):
                    pass
                elif self.Loop(query):
                    pass
                elif self.If(query):
                    pass
                elif self.End(query):
                    pass
                elif self.Assignment(query):
                    pass
                elif self.CuOperator(query):
                    is_cu = True
                elif query == '$':
                    self.EndIf()
                else:
                    print('非法语句：' + query)
                    exit(1)
                # print('解析语句：' + query)
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
            self.loop_or_if_id = self.node_id
            if c_state == 'loop':
                self.loop_id = self.root_id
        elif (self.state == 'loop' or self.state == 'if_branch') and (c_state == 'loop' or c_state == 'if'):
            if c_state == 'if':
                self.state_stack.append([self.loop_or_if_id, self.state, self.out_var, self.in_var, self.oth_branch])
            elif c_state == 'loop':
                self.state_stack.append([self.loop_or_if_id, self.state, self.out_var, self.in_var, self.loop_id])
                self.loop_id = self.node_id
            self.root_id = self.node_id
            self.state = c_state
            self.out_var = self.var_dict.keys()
            self.loop_or_if_id = self.root_id
        elif self.state == 'if' and c_state == 'if_branch':
            self.root_id = self.node_id
            self.state = c_state
        elif self.state == 'if_branch' and c_state == 'end':
            self.state = 'if'
        elif (self.state == 'if' or self.state == 'loop') and c_state == 'end':
            self.root_id = self.node_id
            if len(self.state_stack) == 0:
                self.state = ''
                self.loop_id = 0
            else:
                state_li = self.state_stack[-1]
                if state_li[1] == 'if_branch' and self.state == 'if':
                    self.oth_branch = state_li[4]
                elif state_li[1] == 'loop' and self.state == 'loop':
                    self.loop_id = state_li[4]
                self.in_var = state_li[3]
                self.out_var = state_li[2]
                self.state = state_li[1]
                self.loop_or_if_id = state_li[0]
        elif len(self.state) == 0 and c_state == 'end':
            print('多余括号！')
            exit(1)

    def UpdateVarList(self, v_name, nd_id):
        """
        用于维护变量名列表的函数
        :param v_name: 需要维护的变量名
        :param nd_id: 该变量名对应的最近一次赋值的节点
        :return: 无
        """
        v_name = self.pre + v_name
        var_li = self.var_dict.get(v_name, None)
        if var_li:
            new_li = var_li
            new_li.append(self.graph.nodes[nd_id])
            self.var_dict[v_name] = new_li
        else:
            new_li = list()
            new_li.append(self.graph.nodes[nd_id])
            self.var_dict[v_name] = new_li

    def EndIf(self):
        """
        用于结束if语句
        :return: 无
        """
        if self.state == 'if':
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'IfEnd')
            self.graph.InsertNode(node)
            for l_n in self.leaf_li:
                self.graph.InsertEdge(l_n, self.graph.nodes[self.node_id])
            self.leaf_li.clear()
            self.leaf_li.add(node)
            self.ConnInVar(self.node_id)
            self.StateConvert('end')  # 以if状态下非if型语句解析结束if状态

    def DealInVar(self, v_name, is_new=False):
        """
        处理循环结构和条件结构内部的变量，连接内部用到的外部变量和"收集"内部新建变量
        :param v_name: 变量名
        :param is_new: 是否为新建变量
        :return: 无
        """
        if self.state == 'loop' or self.state == 'if_branch':
            if is_new:
                self.in_var.append(v_name)
            elif v_name in self.out_var:
                var_li = self.var_dict.get(v_name)
                last_use = var_li[-1]
                self.graph.InsertEdge(last_use, self.graph.nodes[self.loop_or_if_id])
                self.graph.InsertEdge(self.graph.nodes[self.loop_or_if_id], self.graph.nodes[self.node_id])
                self.UpdateVarList(v_name, self.loop_or_if_id)

    def ConnInVar(self, e_node):
        """
        连接循环结构和条件结构的内部新建变量到end节点
        :param e_node: end节点
        :return: 无
        """
        for i_n in self.in_var:
            self.UpdateVarList(i_n, e_node)

    def MatchLogicExp(self, m_str):
        """
        对条件语句中的条件进行解析，识别出所含变量，并得到变量名对应的最后一次赋值
        :param m_str: 用于解析的"条件"
        :return: 包含"条件"和变量名最后一次赋值
        """
        match_reg = '[a-zA-Z_]+[a-zA-Z_]*'
        matchObj = re.findall(match_reg, m_str)
        if matchObj:
            v_li = list()
            for v in matchObj:
                if v != 'or' and v != 'and':
                    var_li = self.var_dict.get(v, None)
                    last_use = var_li[-1]
                    v_li.append([v, last_use])
            return v_li
        else:
            return None

    #  用于解析语句的主要函数
    def CreateTensor(self, query):
        """
        解析create语句的函数，对于合法的语句进行建图操作
        :param query: 需要解析的语句
        :return: True 语句合法，False 语句非法
        """
        # 用于匹配的正则表达式
        variable_name_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*)'
        data_list_reg = '[(]([1-9,]?(-1,)?)*([1-9])[)]|[(]([1-9,]?(-1,)?)*(-1)[)]|[(]-1,[)]|[(][1-9],[)]'
        random_reg = '[(][1-9],-1[)]|[(]-1,[1-9][)]|[(][1-9],[1-9][)]'
        create_tensor_reg = f'^CREATE TENSOR {variable_name_reg}[^ ]*( FROM [^ ]+)?( WITH GRAD)?( AS [A-Z]+)?\n$'
        val_info_reg1 = '[1-9]+[.][0-9]+|0[.][0-9]+|[1-9]+[0-9]*|0'
        val_info_reg2 = variable_name_reg  # 暂时考虑使用变量名的要求,待修改
        val_info_reg3 = 'RANDOM[(](.+)[)]'

        # 对读入的字符进行匹配检验是否合法和提取信息
        hasData = False  # 所建的Tensor要存数据
        hasWith = False  # 是否需要记录梯度
        hasAS = 0 # 是否为自定义算子的输入|输出
        hasFrom = 0  # 赋值结点类型
        legal_info = []  # 记录合法的信息
        matchObj = re.match(create_tensor_reg, query)
        if matchObj:
            query = matchObj.group()
            if re.search('WITH', query):
                hasWith = True
            if re.search('AS INPUT', query):
                hasAS = 1
            elif re.search('AS OUTPUT', query):
                hasAS = 2
            T_name = matchObj.group(1)
            if self.var_dict.get(T_name, None):
                return False
            li = query.split(' ')
            if len(T_name) != len(li[2]):
                data = re.search(data_list_reg, li[2])
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
                for i in range(3, len(li) - 1):
                    if 'FROM' == li[i]:
                        j = i + 1
                        from_str = li[j].split('\n')[0]
                if len(from_str) != 0:
                    match1 = re.match(val_info_reg1, from_str)
                    match2 = re.match(val_info_reg2, from_str)
                    match3 = re.match(val_info_reg3, from_str)
                    if match1:
                        value_str = match1.group()
                        if re.search('[.]', value_str):
                            value = float(value_str)
                        else:
                            value = int(value_str)
                        legal_info.append(value)
                        hasFrom = 1
                    elif match3:
                        in_random_str = match3.group(1)
                        type_li = re.findall('[a-zA-z]', in_random_str)
                        if len(re.findall('[(]', in_random_str)) == 2:
                            in_random_reg = f'^({data_list_reg}),({random_reg})'
                            ran_matchObj = re.match(in_random_reg, in_random_str)
                            data_shape = ran_matchObj.group(1)
                            boundary = ran_matchObj.group(2)
                        elif len(re.findall('[(]', in_random_str)) == 1:
                            ran_matchObj = re.search(random_reg, in_random_str)
                            data_shape = '()'
                            boundary = ran_matchObj.group()
                        else:
                            return False
                        legal_info.append([data_shape, boundary, type_li])
                        hasFrom = 3
                    elif match2:
                        t_name = match2.group()
                        if len(t_name) > 30:
                            return False
                        legal_info.append(t_name)
                        hasFrom = 2
                    else:
                        print('dd')
                        return False
        else:
            return False
        # 对上一步的结果进行处理
        self.EndIf()
        if hasWith:
            with_grad = True
        else:
            with_grad = False
        self.node_id += 1
        if hasData:
            node1 = Nd.InstantiationClass(self.node_id, 'CreateTensor', with_grad, data_shape=legal_info[1])
        else:
            node1 = Nd.InstantiationClass(self.node_id, 'CreateTensor', with_grad, data_shape='()')
        self.graph.InsertNode(node1)
        self.leaf_li.add(node1)
        self.graph.InsertEdge(self.graph.nodes[self.root_id], self.graph.nodes[self.node_id])
        if self.graph.nodes[self.root_id] in self.leaf_li:
            self.leaf_li.remove(self.graph.nodes[self.root_id])
        if hasAS == 1:
            self.input.append([legal_info[0], self.node_id])
        elif hasAS == 2:
            self.output.append([legal_info[0], self.node_id])
        if hasFrom != 0:
            self.leaf_li.remove(node1)
            if hasData:
                from_info = legal_info[2]
            else:
                from_info = legal_info[1]
            node1_id = self.node_id
            self.node_id += 1
            if hasFrom == 1:
                node2 = Nd.InstantiationClass(self.node_id, 'Val', with_grad)
                node2.set_val(legal_info[0])
            elif hasFrom == 2:
                node2 = Nd.InstantiationClass(self.node_id, 'Sql', with_grad, t_info=from_info)
            else:
                node2 = Nd.InstantiationClass(self.node_id, 'Random', with_grad,
                                              data_shape=from_info[0], boundary=from_info[1], type=from_info[2])
            self.graph.InsertNode(node2)
            node2_id = self.node_id
            self.node_id += 1
            node3 = Nd.InstantiationClass(self.node_id, 'Assignment', with_grad)
            self.graph.InsertNode(node3)
            self.leaf_li.add(node3)
            self.graph.InsertEdge(self.graph.nodes[node1_id], self.graph.nodes[self.node_id])
            self.graph.InsertEdge(self.graph.nodes[node2_id], self.graph.nodes[self.node_id])
            self.graph.InsertEdge(self.graph.nodes[self.root_id], self.graph.nodes[node2_id])
            self.DealInVar(legal_info[0], True)
        self.UpdateVarList(legal_info[0], self.node_id)
        return True

    def Loop(self, query):
        """
        解析loop语句的函数，对于合法的语句进行建图
        :param query: 需要解析的语句
        :return:True 合法语句，False 非法语句
        """
        loop_reg = 'LOOP ([1-9][0-9]*|true){\n$'
        matchObj = re.match(loop_reg, query)
        if matchObj:
            loop_str = matchObj.group(1)
            if loop_str == 'true':
                condition = True
            else:
                condition = int(loop_str)
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'Loop', condition=condition, loop_id=self.loop_id)
            self.graph.InsertNode(node)
            for l_n in self.leaf_li:
                self.graph.InsertEdge(l_n, self.graph.nodes[self.node_id])
            self.leaf_li.clear()
            self.StateConvert('loop')
            self.EndIf()
            return True
        else:
            return False

    def Break(self, query):
        """
        解析循环语句的break，对于break语句进行相关建图操作
        :param query: 需要解析的语句
        :return: True 合法语句 False 非法语句
        """
        if self.loop_id == 0:
            return False
        break_reg = 'BREAK'
        matchObj = re.match(break_reg, query)
        if matchObj:
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'Break', loop_id=self.loop_id)
            self.graph.InsertNode(node)
            for l_n in self.leaf_li:
                self.graph.InsertEdge(l_n, self.graph.nodes[self.node_id])
            self.leaf_li.clear()
            self.leaf_li.add(node)
            return True
        else:
            return False

    def If(self, query):
        """
        解析if语句的函数，对于合法的语句进行建图
        :param query: 需要解析的语句
        :return:True 合法语句，False 非法语句
        """
        con_reg = '[(][a-zA-Z0-9_=!<> ]+[)]'
        if_reg = 'IF '+con_reg+'{\n'  # 所有关于if的正则，目前未对条件进行约束，待修改
        elif_reg = 'ELIF '+con_reg+'{\n'
        else_reg = 'ELSE{\n'
        matchObj_if = re.match(if_reg, query)
        matchObj_elif = re.match(elif_reg, query)
        matchObj_else = re.match(else_reg, query)
        if matchObj_if:
            self.EndIf()
            if_str = matchObj_if.group()
            condition = re.search(con_reg, if_str).group()
            var_li = self.MatchLogicExp(condition)
            if not var_li:
                return False
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'If')
            self.graph.InsertNode(node)
            for l_n in self.leaf_li:
                self.graph.InsertEdge(l_n, self.graph.nodes[self.node_id])
            self.leaf_li.clear()
            self.StateConvert('if')
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'IfBranch')
            self.graph.InsertNode(node)
            self.graph.InsertEdge(self.graph.nodes[self.root_id], self.graph.nodes[self.node_id],
                                  condition, need_var=var_li)
            self.StateConvert('if_branch')
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'IfBranch')
            self.graph.InsertNode(node)
            self.leaf_li.add(node)
            self.graph.InsertEdge(self.graph.nodes[self.loop_or_if_id], self.graph.nodes[self.node_id],
                                  'T'+'$'+condition, need_var=var_li)
            self.oth_branch = self.node_id
            return True
        elif matchObj_elif:
            if self.state == 'if':
                if_str = matchObj_elif.group()
                condition = re.search(con_reg, if_str).group()
                var_li = self.MatchLogicExp(condition)
                print(var_li)
                if not var_li:
                    return False
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'IfBranch')
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.graph.nodes[self.oth_branch], self.graph.nodes[self.node_id],
                                      condition, need_var=var_li)
                if self.graph.nodes[self.oth_branch] in self.leaf_li:
                    self.leaf_li.remove(self.graph.nodes[self.oth_branch])
                self.StateConvert('if_branch')
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'IfBranch')
                self.graph.InsertNode(node)
                self.leaf_li.add(node)
                self.graph.InsertEdge(self.graph.nodes[self.oth_branch], self.graph.nodes[self.node_id],
                                      'T'+'$'+condition, need_var=var_li)
                self.oth_branch = self.node_id
                return True
            else:
                return False
        elif matchObj_else:
            if self.state == 'if':
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'IfBranch')
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.graph.nodes[self.oth_branch], self.graph.nodes[self.node_id])
                if self.graph.nodes[self.oth_branch] in self.leaf_li:
                    self.leaf_li.remove(self.graph.nodes[self.oth_branch])
                self.StateConvert('if_branch')
                return True
            else:
                return False
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
            self.EndIf()
            if self.state == 'loop':
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'LoopEnd', loop_id=self.loop_id)
                self.graph.InsertNode(node)
                for l_n in self.leaf_li:
                    self.graph.InsertEdge(l_n, self.graph.nodes[self.node_id])
                self.leaf_li.clear()
                self.ConnInVar(self.node_id)
                self.graph.InsertEdge(self.graph.nodes[self.node_id], self.graph.nodes[self.loop_id])
                self.leaf_li.add(node)
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
        variable_name_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*)'
        ass_reg = f'^{variable_name_reg} = (.+)\n$'
        sql_reg = 'SELECT ([a-zA-Z_]+[a-zA-Z0-9_]*) FROM ([a-zA-Z_]+[a-zA-Z0-9_]*)( WHERE .+)?'
        matchObj = re.match(ass_reg, query)
        if matchObj:
            exp = matchObj.group()
            v_name = matchObj.group(1)
            ass_exp = matchObj.group(2)
            # self.node_id += 1
            # p = A_e.Analyze_expression(A_e.Pretreatment(exp), self.node_id)
            # if p:
            #     g = p[0]
            #     g_in = p[1]
            #     g_out = p[2]
            #     print(g_in)
            #     print(g_out)
            #     if not g_out:
            #         return False
            #     self.graph.Merge([g.nodes, g.edges])
            #     for in_v in g_in:
            #         var_li = self.var_dict.get(in_v[0], None)
            #         if var_li:
            #             last_use = var_li[-1]
            #             self.graph.InsertEdge(last_use, self.graph.nodes[in_v[1]])
            #         else:
            #             print('cc')
            #             return False
            #     e_node_id = g_out.GetId()
            #     self.node_id = e_node_id
            sql_matchObj = re.match(sql_reg, ass_exp)
            if sql_matchObj:
                t_info = 'C'+'$'+sql_matchObj.group(1)+'$'+sql_matchObj.group(2)
                hasWhere = re.search('WHERE (.+)', sql_matchObj.group())
                if hasWhere:
                    t_info = 'W'+'$'+t_info+'$'+hasWhere.group(1)
                self.node_id += 1
                e_node = Nd.InstantiationClass(self.node_id, 'Sql', t_info=t_info)
                e_node_id = self.node_id
                self.graph.InsertNode(e_node)
                self.graph.InsertEdge(self.graph.nodes[self.root_id], self.graph.nodes[self.node_id])
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
            self.graph.InsertEdge(last_use, ass_n)
            if last_use in self.leaf_li:
                self.leaf_li.remove(last_use)
        else:
            self.DealInVar(v_name, True)
            self.node_id += 1
            node_l = Nd.InstantiationClass(self.node_id, 'CreateTensor', data_shape='()')
            self.graph.InsertNode(node_l)
            node_l_id = self.node_id
            self.node_id += 1
            ass_n = Nd.InstantiationClass(self.node_id, 'Assignment')
            self.graph.InsertNode(ass_n)
            self.graph.InsertEdge(self.graph.nodes[self.root_id], self.graph.nodes[node_l_id])
            if self.graph.nodes[self.root_id] in self.leaf_li:
                self.leaf_li.remove(self.graph.nodes[self.root_id])
            self.graph.InsertEdge(self.graph.nodes[node_l_id], ass_n)
        self.graph.InsertEdge(self.graph.nodes[e_node_id], ass_n)
        self.leaf_li.add(ass_n)
        self.UpdateVarList(v_name, self.node_id)
        return True

    # 自定义算子需要递归使用解析器，所以要使用一些特殊方法
    @staticmethod
    def CuEnd(query):
        """
        用于识别\n}结束自定义算子语句结构
        :param query:
        :return: False 结束，True 不结束
        """
        end_reg = '\n} (AS RELATION)?'
        matchObj = re.match(end_reg, query)
        if matchObj:
            return False
        else:
            return True

    def CuOperator(self, query):
        """
        用于判断是否为自定义语句，开始自定义算子语句结构
        :param query:
        :return: True 合法语句，False 非法语句
        """
        self.EndIf()
        c_o_reg = 'OPERATOR ([a-zA-Z_]+[a-zA-Z0-9_]*){\n'
        matchObj = re.match(c_o_reg, query)
        if matchObj:
            return True
        else:
            return False

    def GetGInfo(self):
        g_info = (self.input, self.output, self.graph, self.var_dict)
        return g_info


if __name__ == '__main__':
    # create语句测试
    # create_test = list()
    # create_test.append('CREATE TENSOR LR(1,4)\n')
    # create_test.append('CREATE TENSOR LL2(-1,4)\n')
    # create_test.append('CREATE TENSOR Ll(3,-1)\n')
    # create_test.append('CREATE TENSOR A_(-1,)\n')
    # create_test.append('CREATE TENSOR Ba4S(3,)\n')
    # create_test.append('CREATE TENSOR lR(1,4) FROM 7\n')
    # create_test.append('CREATE TENSOR _LR(4) FROM 0.04\n')
    # create_test.append('CREATE TENSOR LR1(1,4) FROM RANDOM((2,4))\n')
    # create_test.append('CREATE TENSOR R1(1,4) FROM RANDOM((-1,2,3),(2,4))\n')
    # create_test.append('CREATE TENSOR L1(1,4) FROM RANDOM((2,4),uniform)\n')
    # create_test.append('CREATE TENSOR LR2(1,4) FROM User\n')
    # create_test.append('CREATE TENSOR LR3(1,4) FROM User WITH GRAD\n')
    # testPar = Parser(create_test)
    # testPar()
    #  loop语句测试
    loop_test = list()
    loop_test.append('LOOP 108{\n')
    loop_test.append('CREATE TENSOR LL2(-1,4)\n')
    loop_test.append('CREATE TENSOR A_(-1,)\n')
    loop_test.append('LOOP true{\n')
    loop_test.append('CREATE TENSOR _LR(1,4) FROM 0.04\n')
    loop_test.append('\n}')
    loop_test.append('\n}')
    testPar = Parser(loop_test)
    testPar()
    # if语句测试
    # if_test = list()
    # if_test.append('CREATE TENSOR a(-1,3) FROM 287\n')
    # if_test.append('IF (a<108){\n')
    # if_test.append('CREATE TENSOR LL2(-1,4)\n')
    # if_test.append('CREATE TENSOR A_(-1,)\n')
    # if_test.append('IF (a<18){\n')
    # if_test.append('CREATE TENSOR _LR(1,4) FROM 0.04\n')
    # if_test.append('\n}')
    # if_test.append('\n}')
    # if_test.append('ELIF (205>a and a>108){\n')
    # if_test.append('CREATE TENSOR _LR(1,4) FROM 0.04\n')
    # if_test.append('\n}')
    # if_test.append('ELSE{\n')
    # if_test.append('CREATE TENSOR LL2(-1,4)\n')
    # if_test.append('\n}')
    # if_test.append('CREATE TENSOR LR(1,4) FROM User\n')
    # testPar = Parser(if_test)
    # testPar()
    # 表达式测试
    # exp_test = list()
    # exp_test.append('CREATE TENSOR x(-1,3,4,-1,3)\n')
    # exp_test.append('CREATE TENSOR y(-1,2,6,-1,7)\n')
    # exp_test.append('CREATE TENSOR z(-1,3,-1,4,-1)\n')
    # exp_test.append('CREATE TENSOR a(-1,3,4)\n')
    # exp_test.append('CREATE TENSOR b(3,-1)\n')
    # exp_test.append('CREATE TENSOR c(-1,)\n')
    # exp_test.append('CREATE TENSOR e(-1,-1)\n')
    # exp_test.append('a = x + y / z - x * x\n')
    # exp_test.append('b = LOG(x)\n')
    # exp_test.append('c = MATMUL(y , z) WITH GRAD\n')
    # exp_test.append('e = SELECT user_name FROM USER WHERE true\n')
    # test_par = Parser(exp_test)
    # test_par()
