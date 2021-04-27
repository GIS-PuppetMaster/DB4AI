import Digraph as DG
import Nodes as Nd
import re
import copy
import Analyze_expression as A_e
import json
import os
import pickle
from Executor import Executor


class Parser:
    def __init__(self, queries: list):
        self.queries = queries
        self.var_dict = dict()
        self.graph = DG.Graph()
        self.node_id = 0
        # 记录”状态“，用于特殊语句的解析使用
        self.root_id = 0
        self.state = ''
        self.state_stack = list()
        self.loop_or_if_id = 0
        self.loop_id = 0
        self.out_var = dict()
        self.oth_branch = 0
        self.extra_pop_num = 0
        self.branches = list()
        #  用于自定义算子使用的特殊域
        self.input = list()
        self.operator = ''
        self.isCu = False

    def __call__(self, **kwargs):
        """
        类的call方法，使用类中的语句解析方法解析语句列表，进行建图
        :param kwargs: 暂时未使用
        """
        root = Nd.InstantiationClass(self.node_id, 'Root', self.branches)
        self.graph.InsertNode(root)
        self.branches.append(0)
        self.queries[-1] = self.queries[-1] + '\n'
        self.queries.append('$')
        for query in self.queries:
            query = query.lstrip()
            if len(query) == 0 or query[0] == '#':
                continue
            if self.CreateTensor(query):
                pass
            elif self.Loop(query):
                pass
            elif self.Assignment(query):
                pass
            elif self.If(query):
                pass
            elif self.Break(query):
                pass
            elif self.End(query):
                pass
            elif self.CuOperator(query):
                pass
            elif query == '$':
                self.EndIf()
            else:
                self.graph.Show()
                raise Exception('非法语句：' + query)
        self.graph.Show()
        return self.graph

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
            self.out_var = copy.deepcopy(self.var_dict)
            self.loop_or_if_id = self.node_id
            if c_state == 'loop':
                self.loop_id = self.root_id
                self.branches.append(self.root_id)
        elif (self.state == 'loop' or self.state == 'if_branch') and (c_state == 'loop' or c_state == 'if'):
            self.root_id = self.node_id
            if self.state == 'if_branch':
                self.state_stack.append([self.loop_or_if_id, self.state, copy.deepcopy(self.out_var),
                                         self.branches.copy(), self.oth_branch, self.extra_pop_num])
            elif self.state == 'loop':
                self.state_stack.append([self.loop_or_if_id, self.state, copy.deepcopy(self.out_var),
                                         self.branches.copy(), self.loop_id])
            if c_state == 'loop':
                self.loop_id = self.node_id
                self.branches.append(self.root_id)
            self.state = c_state
            self.out_var = copy.deepcopy(self.var_dict)
            self.loop_or_if_id = self.root_id
        elif self.state == 'if' and c_state == 'if_branch':
            self.root_id = self.node_id
            self.state = c_state
            self.branches.append(self.root_id)
        elif self.state == 'if_branch' and c_state == 'end':
            self.state = 'if'
            del self.branches[-1]
        elif (self.state == 'if' or self.state == 'loop') and c_state == 'end':
            self.root_id = self.node_id
            if len(self.state_stack) == 0:
                if self.state == 'loop':
                    self.branches.pop(-1)
                elif self.state == 'if':
                    while self.extra_pop_num != 0:
                        self.branches.pop(-1)
                        self.extra_pop_num += -1
                self.state = ''
            else:
                state_li = self.state_stack.pop(-1)
                if state_li[1] == 'if_branch':
                    self.extra_pop_num = state_li[5]
                    self.oth_branch = state_li[4]
                elif state_li[1] == 'loop':
                    self.loop_id = state_li[4]
                self.branches = state_li[3]
                self.out_var = state_li[2]
                self.state = state_li[1]
                self.loop_or_if_id = state_li[0]
        elif len(self.state) == 0 and c_state == 'end':
            if self.isCu:
                # self.graph.Show()
                output = self.graph.GetNoOutNodes()
                if not output:
                    output = copy.copy(self.graph.nodes[self.node_id])
                self.AddUserOperator(output, self.input, self.graph, self.operator)
                self.Reset()
            else:
                raise Exception('多余括号！')

    def UpdateVarList(self, v_name, nd_id):
        """
        用于维护变量名列表的函数
        :param v_name: 需要维护的变量名
        :param nd_id: 该变量名对应的最近一次赋值的节点
        :return: 无
        """
        v_name = v_name
        var_li = self.var_dict.get(v_name, None)
        if var_li:
            var_li.append(nd_id)
            self.var_dict[v_name] = var_li
        else:
            new_li = list()
            new_li.append(nd_id)
            self.var_dict[v_name] = new_li

    def EndIf(self):
        """
        用于结束if语句
        :return: 无
        """
        if self.state == 'if':
            self.node_id += 1
            self.StateConvert('end')  # 以if状态下非if型语句解析结束if状态
            node = Nd.InstantiationClass(self.node_id, 'IfEnd', self.branches)
            for l_n in self.graph.GetNoOutNodes().copy():
                if isinstance(l_n, Nd.IfBranch):
                    com_branches = l_n.branches.copy()
                    com_branches.pop(-1)
                    if node.branches != com_branches:
                        continue
                self.graph.InsertEdge(l_n, node)
            self.branches.append(self.root_id)
            self.graph.InsertNode(node)

    def DealInVar(self, v_name):
        """
        处理循环结构和条件结构内部的变量，连接内部用到的外部变量和"收集"内部新建变量
        :param v_name: 变量名
        :return: 无
        """
        if (self.state == 'loop' or self.state == 'if_branch') and v_name in self.out_var.keys():
            var_li = self.out_var.get(v_name)
            last_use = var_li[-1]
            if self.graph.nodes[last_use].branches == self.graph.nodes[self.loop_or_if_id].branches:
                self.graph.InsertEdge(self.graph.nodes[last_use], self.graph.nodes[self.loop_or_if_id])

    def MatchLogicExp(self, m_str):
        """
        对条件语句中的条件进行解析，识别出所含变量，并得到变量名对应的最后一次赋值
        :param m_str: 用于解析的"条件"
        :return: 包含"条件"和变量名最后一次赋值
        """
        match_reg = '[a-zA-Z_]+[a-zA-Z_]*'
        match_obj = re.findall(match_reg, m_str)
        if match_obj:
            v_li = list()
            for v in match_obj:
                if v != 'or' and v != 'and':
                    var_li = self.var_dict.get(v, None)
                    last_use = var_li[-1]
                    v_li.append([v, self.graph.nodes[last_use]])
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
        data_reg = '[+-]?([1-9][0-9]*|0)(.[0-9]+)?|[+-]?([1-9][0-9]*(.[0-9]+)?|0.[0-9]+)e([+-]?[1-9][0-9]*|0)' \
                   '|[a-zA-Z_]+[a-zA-Z0-9_]*'
        variable_name_reg = '[a-zA-Z_]+[a-zA-Z0-9_]*'
        data_shape_reg = '[(](([1-9][0-9]*,|-1,|[a-zA-Z_]+[a-zA-Z0-9_]*,))+([1-9][0-9]*|-1|[a-zA-Z_]+[a-zA-Z0-9_]*)?[)]'
        random_reg = '[(]([+-]?([1-9][0-9]*|0)(.[0-9]+)?' \
                     '|[+-]?([1-9][0-9]*(.[0-9]+)?|0.[0-9]+)e([+-]?[1-9][0-9]*|0)|[a-zA-Z_]+[a-zA-Z0-9_]*)' \
                     ',([+-]?([1-9][0-9]*|0)(.[0-9]+)?|[+-]?([1-9][0-9]*(.[0-9]+)?|0.[0-9]+)e([+-]?[1-9][0-9]*|0)' \
                     '|[a-zA-Z_]+[a-zA-Z0-9_]*)[)]'
        create_tensor_reg = f'^(CREATE|create)[ \t]*(TENSOR|tensor)[ \t]*({variable_name_reg}[ \t]*(.+?))' \
                            f'([ \t]*(FROM|from)[ \t]*(.+?))?([ \t]*(WITH|with)[ \t]*(GRAD|grad))?\n$'
        val_info_reg1 = '[+-]?([1-9][0-9]*|0)(.[0-9]+)?'
        val_info_reg2 = '(SQL|sql)[(](.+)[)]'  # 暂时考虑使用变量名的要求,待修改
        val_info_reg3 = f'^(RANDOM|random)([(]({data_shape_reg}),({random_reg})(,\'[a-zA-Z]+\')?[)])'

        # 对读入的字符进行匹配检验是否合法和提取信息
        hasWith = False  # 是否需要记录梯度
        from_str = ''  # 赋值结点类型
        from_type = 0
        legal_info = []  # 记录合法的信息
        match_obj = re.match(create_tensor_reg, query)
        if match_obj:
            query = re.sub('[ \t]+', '', match_obj.group())
            if re.search('WITH|with', query):
                hasWith = True
            fromObj = re.search('(FROM|from)(.+)(with|WITH)|(FROM|from)(.+)', query)
            if fromObj and hasWith:
                from_str = fromObj.group(2)
            elif fromObj:
                from_str = fromObj.group(5)
            T_info = re.sub('[ \t]+', '', match_obj.group(3))
            T_name = re.match(f'^{variable_name_reg}', T_info).group()
            if self.var_dict.get(T_name, None):
                raise Exception('重复创建张量：' + T_name + '，语句为：' + query)
            data = re.search(data_shape_reg, T_info)
            if data:
                data_shape = data.group()
                legal_info.append(T_name)
                legal_info.append(data_shape)
            else:
                raise Exception('张量data_shape错误：' + T_info + '，语句为：' + query)
            if len(from_str) != 0:
                match1 = re.match(val_info_reg1, from_str)
                match2 = re.match(val_info_reg2, from_str)
                match3 = re.match(val_info_reg3, from_str)
                match4 = re.match(f'^(ZEROS|zeros)[(]({data_shape_reg})[)]', from_str)
                match5 = re.match(f'^(ONES|ones)[(]({data_shape_reg})[)]', from_str)
                match6 = re.match(f'^(FULL|full)([(]({data_shape_reg}),({data_reg})[)])', from_str)
                if match1:
                    value_str = match1.group()
                    if re.search('[.]', value_str):
                        value = float(value_str)
                    else:
                        value = int(value_str)
                    legal_info.append(value)
                    from_type = 1
                elif match3:
                    in_random_str = match3.group(2)
                    type_search_Obj = re.search(',\'([a-zA-z]+)\'[)]', in_random_str)
                    if type_search_Obj:
                        type = type_search_Obj.group(1)
                    else:
                        type = ''
                    ran_match_obj = re.match('[(]([(].+[)]),([(].+[)]).*[)]', in_random_str)
                    if ran_match_obj:
                        data_shape = ran_match_obj.group(1)
                        boundary = ran_match_obj.group(2)
                    else:
                        raise Exception('RANDOM信息错误：' + in_random_str + '，语句为：' + query)
                    legal_info.append([data_shape, boundary, type])
                    from_type = 3
                elif match2:
                    t_info = match2.group(1)
                    legal_info.append(t_info)
                    from_type = 2
                elif match4:
                    legal_info.append(match4.group(2))
                    from_type = 4
                elif match5:
                    legal_info.append(match5.group(2))
                    from_type = 5
                elif match6:
                    in_full_str = match6.group(2)
                    full_match_obj = re.match('[(]([(].+[)]),(.+)[)]', in_full_str)
                    if full_match_obj:
                        data_shape = full_match_obj.group(1)
                        num = full_match_obj.group(2)
                        legal_info.append([data_shape, num])
                        from_type = 6
                else:
                    raise Exception('FROM信息错误：' + from_str + '，语句为：' + query)
        else:
            return False
        # 对上一步的结果进行处理
        self.EndIf()
        if hasWith:
            with_grad = True
        else:
            with_grad = False
        self.node_id += 1
        node1 = Nd.InstantiationClass(self.node_id, 'CreateTensor', self.branches, with_grad, data_shape=legal_info[1],
                                      var=[legal_info[0]])
        self.graph.InsertNode(node1)
        if self.isCu and self.root_id == 0:
            pass
        else:
            self.graph.InsertEdge(self.graph.nodes[self.root_id], self.graph.nodes[self.node_id])
        if from_type != 0:
            from_info = legal_info[2]
            node1_id = self.node_id
            self.node_id += 1
            if from_type == 1:
                node2 = Nd.InstantiationClass(self.node_id, 'Val', self.branches, with_grad,
                                              var=['@' + str(self.node_id)], val=legal_info[2])
            elif from_type == 2:
                node2 = Nd.InstantiationClass(self.node_id, 'Sql', self.branches, with_grad, t_info=from_info,
                                              var=['@' + str(self.node_id)])
            elif from_type == 3:
                node2 = Nd.InstantiationClass(self.node_id, 'Random', self.branches, with_grad, data_shape=from_info[0],
                                              boundary=from_info[1], type=from_info[2], var=['@' + str(self.node_id)])
                match_obj_d = re.findall(r'[a-zA-Z_]+[a-zA-Z0-9_]*', from_info[0])
                match_obj_b = re.findall(r'', from_info[1])
                if len(match_obj_d) != 0:
                    d_has_var = True
                    d_var = dict()
                    for obj in match_obj_d:
                        d_var[obj] = None
                else:
                    d_var = dict()
                    d_has_var = False
                if len(match_obj_b) != 0:
                    b_var = dict()
                    b_has_var = True
                    for obj in match_obj_b:
                        b_var[obj] = None
                else:
                    b_var = dict()
                    b_has_var = False
                node2.handle_include_var(b_has_var=b_has_var, d_has_var=d_has_var, b_var=b_var, d_var=d_var)
            elif from_type == 4:
                node2 = Nd.InstantiationClass(self.node_id, 'Zeros', self.branches, with_grad, data_shape=from_info,
                                              var=['@' + str(self.node_id)])
            elif from_type == 5:
                node2 = Nd.InstantiationClass(self.node_id, 'Ones', self.branches, with_grad, data_shape=from_info,
                                              var=['@' + str(self.node_id)])
            else:
                node2 = Nd.InstantiationClass(self.node_id, 'Full', self.branches, with_grad, data_shape=from_info[0],
                                              var=['@' + str(self.node_id)], num=from_info[1])
            self.graph.InsertNode(node2)
            node2_id = self.node_id
            self.UpdateVarList('@' + str(self.node_id), self.node_id)
            self.node_id += 1
            node3 = Nd.InstantiationClass(self.node_id, 'Assignment', self.branches, with_grad,
                                          var_li=[legal_info[0], '@' + str(node2_id)])
            self.graph.InsertNode(node3)
            self.graph.InsertEdge(self.graph.nodes[node1_id], self.graph.nodes[self.node_id])
            self.graph.InsertEdge(self.graph.nodes[node2_id], self.graph.nodes[self.node_id])
            if self.isCu and self.root_id == 0:
                pass
            else:
                self.graph.InsertEdge(self.graph.nodes[self.root_id], self.graph.nodes[node2_id])
        self.UpdateVarList(legal_info[0], self.node_id)
        return True

    def Loop(self, query):
        """
        解析loop语句的函数，对于合法的语句进行建图
        :param query: 需要解析的语句
        :return:True 合法语句，False 非法语句
        """
        loop_reg = '(LOOP|loop)[ \t]*[(]([1-9][0-9]*|TRUE|true|[a-zA-Z_]+[a-zA-Z0-9_]*)[)]{\n$'
        match_obj = re.match(loop_reg, query)
        if match_obj:
            self.EndIf()
            loop_str = match_obj.group(2)
            if loop_str == 'true' or loop_str == 'TRUE':
                condition = True
            elif re.search('[1-9][0-9]*', loop_str):
                condition = int(loop_str)
            else:
                condition = loop_str
            self.node_id += 1
            root_id = self.root_id
            com_branches = self.branches.copy()
            self.StateConvert('loop')
            node = Nd.InstantiationClass(self.node_id, 'Loop', self.branches, condition=condition, loop_id=self.loop_id)
            for l_n in self.graph.GetNoOutNodes().copy():
                if isinstance(l_n, Nd.IfBranch):
                    continue
                elif l_n.branches == com_branches:
                    self.graph.InsertEdge(l_n, node)
            if len(node.in_edges) == 0:
                self.graph.InsertEdge(self.graph.nodes[root_id], node)
            self.graph.InsertNode(node)
            return True
        else:
            return False

    def Break(self, query):
        """
        解析循环语句的break，对于break语句进行相关建图操作
        :param query: 需要解析的语句
        :return: True 合法语句 False 非法语句
        """
        break_reg = 'BREAK\n$|break\n$'
        match_obj = re.match(break_reg, query)
        if match_obj:
            if self.loop_id == 0:
                raise Exception('非loop内使用break：')
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'Break', self.branches, loop_id=self.loop_id)
            for l_n in self.graph.GetNoOutNodes().copy():
                if l_n.branches == node.branches:
                    self.graph.InsertEdge(l_n, node)
            self.graph.InsertNode(node)
            return True
        else:
            return False

    def If(self, query):
        """
        解析if语句的函数，对于合法的语句进行建图
        :param query: 需要解析的语句
        :return:True 合法语句，False 非法语句
        """
        con_reg = '[a-zA-Z0-9_=!<> ]+'
        if_reg = '(IF|if)[ \t]*[(](.+)[)]{\n$'  # 所有关于if的正则，目前未对条件进行约束，待修改
        elif_reg = '(ELIF|elif)[ \t]*[(](.+)[)]{\n$'
        else_reg = '(ELSE|else)[ \t]*{\n$'
        match_obj_if = re.match(if_reg, query)
        match_obj_elif = re.match(elif_reg, query)
        match_obj_else = re.match(else_reg, query)
        if match_obj_if:
            self.EndIf()
            if_str = match_obj_if.group(2)
            condition = re.search(con_reg, if_str).group()
            if condition != 'true' and condition != 'TRUE':
                var_li = self.MatchLogicExp(condition)
                if not var_li:
                    raise Exception('逻辑语句拼写错误或使用未创建张量：' + condition + '，语句为：' + query)
            else:
                var_li = []
            self.node_id += 1
            node = Nd.InstantiationClass(self.node_id, 'If', self.branches)
            for l_n in self.graph.GetNoOutNodes().copy():
                if isinstance(l_n, Nd.IfBranch):
                    continue
                elif l_n.branches == self.branches:
                    self.graph.InsertEdge(l_n, node)
            if len(node.in_edges) == 0:
                self.graph.InsertEdge(self.graph.nodes[self.root_id], node)
            self.graph.InsertNode(node)
            self.StateConvert('if')
            self.node_id += 1
            branches = self.branches.copy()
            self.StateConvert('if_branch')
            node = Nd.InstantiationClass(self.node_id, 'IfBranch', self.branches)
            self.graph.InsertNode(node)
            self.graph.InsertEdge(self.graph.nodes[self.loop_or_if_id], self.graph.nodes[self.node_id],
                                  condition, need_var=var_li)
            self.node_id += 1
            branches.append(self.node_id)
            node = Nd.InstantiationClass(self.node_id, 'IfBranch', branches)
            self.graph.InsertNode(node)
            self.graph.InsertEdge(self.graph.nodes[self.loop_or_if_id], self.graph.nodes[self.node_id],
                                  'T' + '$' + condition, need_var=var_li)
            self.oth_branch = self.node_id
            return True
        elif match_obj_elif:
            if self.state == 'if':
                if_str = match_obj_elif.group(2)
                condition = re.search(con_reg, if_str).group()
                var_li = self.MatchLogicExp(condition)
                if not var_li:
                    raise Exception('逻辑语句拼写错误或使用未创建张量：' + condition + '，语句为：' + query)
                self.node_id += 1
                self.branches.append(self.oth_branch)
                branches = self.branches.copy()
                self.StateConvert('if_branch')
                node = Nd.InstantiationClass(self.node_id, 'IfBranch', self.branches)
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.graph.nodes[self.oth_branch], self.graph.nodes[self.node_id],
                                      condition, need_var=var_li)
                self.node_id += 1
                branches.append(self.node_id)
                node = Nd.InstantiationClass(self.node_id, 'IfBranch', branches)
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.graph.nodes[self.oth_branch], self.graph.nodes[self.node_id],
                                      'T' + '$' + condition, need_var=var_li)
                self.oth_branch = self.node_id
                self.extra_pop_num += 1
                return True
            else:
                raise Exception('elif在使用if前使用，语句为：' + query)
        elif match_obj_else:
            if self.state == 'if':
                self.node_id += 1
                self.branches.append(self.oth_branch)
                self.StateConvert('if_branch')
                node = Nd.InstantiationClass(self.node_id, 'IfBranch', self.branches)
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.graph.nodes[self.oth_branch], self.graph.nodes[self.node_id])
                self.oth_branch = 0
                self.extra_pop_num += 1
                return True
            else:
                raise Exception('else在使用if前使用')
        else:
            return False

    def End(self, query):
        """
        识别\n}，结束当前语句结构
        :param query: 需要解析的语句
        :return:True 合法语句，False 非法语句
        """
        end_reg = '}\n'
        match_obj = re.match(end_reg, query)
        if match_obj:
            self.EndIf()
            if self.state == 'loop':
                self.node_id += 1
                self.StateConvert('end')
                node = Nd.InstantiationClass(self.node_id, 'LoopEnd', self.branches, loop_id=self.loop_id)
                for l_n in self.graph.GetNoOutNodes().copy():
                    if isinstance(l_n, Nd.IfBranch):
                        continue
                    self.graph.InsertEdge(l_n, node)
                self.branches.append(self.root_id)
                self.graph.InsertNode(node)
                self.graph.InsertEdge(self.graph.nodes[self.node_id], self.graph.nodes[self.loop_id])
            elif self.oth_branch == 0 and self.state == 'if_branch':
                self.StateConvert('end')
                self.EndIf()
            else:
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
        ass_reg1 = f'^{variable_name_reg} = (SQL|sql)[(](.+)[)]\n$'
        ass_reg2 = f'^(SELECT|select)[ \t]+(.+?)([ \t]+(AS|as)[ \t]+(.+?))?' \
                   f'([ \t]+(FROM|from)[ \t]+(.+?))?([ \t]+(WITH|with)[ \t]+(GRAD|grad))?\n$'
        match_obj1 = re.match(ass_reg1, query)
        match_obj2 = re.match(ass_reg2, query)
        slice_info = list()
        with_grad = False
        if match_obj1:
            self.EndIf()
            v_name = match_obj1.group(1)
            search_exp = match_obj1.group(2)
            self.node_id += 1
            e_node = Nd.InstantiationClass(self.node_id, 'Sql', self.branches, t_info=search_exp,
                                           var=['@' + str(self.node_id)])
            self.graph.InsertNode(e_node)
            self.graph.InsertEdge(self.graph.nodes[self.root_id], e_node)
            r_var = '@' + str(e_node.id)
            self.UpdateVarList(r_var, e_node.id)
        elif match_obj2:
            self.EndIf()
            v_name = match_obj2.group(5)
            var_str = match_obj2.group(8)
            if v_name:
                match_obj = re.match(f'^{variable_name_reg}\[(.+)]', v_name)
                if match_obj:
                    v_name = match_obj.group(1)
                    slice_info = match_obj.group(2).split(',')
            else:
                v_name = '$'
            as_replace = dict()
            if var_str is not None:
                var_info = list(map(lambda x: x.strip(), var_str.split(',')))
                for v_i in var_info:
                    v_i = re.sub('[ \t]+','',v_i)
                    as_obj = re.search('(.+?)AS|as(.+?)', v_i)
                    if as_obj:
                        as_replace[as_obj.group(2)] = as_obj.group(1)
            exp = v_name + ' = ' + match_obj2.group(2)
            if re.search('(WITH|with)[ \t]+(GRAD|grad)', query):
                exp = exp + ' WITH GRAD'
                with_grad = True
            self.node_id += 1
            branches = self.branches.copy()
            p = A_e.analyze_expression(exp, self.node_id, branches, as_replace)
            g = p[0]
            g_in = p[1]
            g_out = p[2]
            if p[0] and p[1] and p[2]:
                self.graph.Merge([g[0], g[1]])
                for in_v in g_in:
                    if isinstance(in_v[1], Nd.Loop) or isinstance(in_v[1], Nd.If):
                        for l_n in self.graph.GetNoOutNodes().copy():
                            if isinstance(l_n, Nd.IfBranch):
                                continue
                            elif l_n.branches == branches:
                                self.graph.InsertEdge(l_n, in_v[1])
                        if len(in_v[1].in_edges) == 0:
                            self.graph.InsertEdge(self.graph.nodes[self.root_id], in_v[1])
                    else:
                        var_li = self.var_dict.get(in_v[0], None)
                        if var_li:
                            last_use = var_li[-1]
                            if self.graph.nodes[last_use].branches == in_v[1].branches:
                                self.graph.InsertEdge(self.graph.nodes[last_use], in_v[1])
                            else:
                                self.graph.InsertEdge(self.graph.nodes[self.root_id], in_v[1])
                        elif isinstance(in_v[1], Nd.Val):
                            self.graph.InsertEdge(self.graph.nodes[self.root_id], in_v[1])
                        else:
                            raise Exception('表达式使用未创建张量：' + in_v[0] + '，语句为：' + query)
                e_node = g_out
                self.node_id = self.node_id + len(g[0]) - 1
            else:
                raise Exception('右侧表达式拼写错误，语句为：' + query)
        else:
            return False
        if v_name != '$':
            r_var = e_node.get_vars()[0]
            self.UpdateVarList(r_var, e_node.id)
            var_li = self.var_dict.get(v_name, None)
            if var_li:
                self.node_id += 1
                ass_n = Nd.InstantiationClass(self.node_id, 'Assignment', self.branches, var_li=[v_name, r_var], with_grad=with_grad)
                self.graph.InsertNode(ass_n)
                ass_n.slice = slice_info
                last_use = var_li[-1]
                if self.graph.nodes[last_use].branches == self.branches:
                    self.graph.InsertEdge(self.graph.nodes[last_use], ass_n)
            else:
                self.node_id += 1
                node_l = Nd.InstantiationClass(self.node_id, 'CreateTensor', self.branches, data_shape=None, var=v_name)
                self.graph.InsertNode(node_l)
                self.UpdateVarList(v_name, self.node_id)
                self.node_id += 1
                ass_n = Nd.InstantiationClass(self.node_id, 'Assignment', self.branches, var_li=[v_name, r_var], with_grad=with_grad)
                self.graph.InsertNode(ass_n)
                ass_n.slice = slice_info
                if self.isCu and self.root_id == 0:
                    pass
                else:
                    self.graph.InsertEdge(self.graph.nodes[self.root_id], node_l)
                self.graph.InsertEdge(node_l, ass_n)
            self.UpdateVarList(v_name, self.node_id)
            self.graph.InsertEdge(e_node, ass_n)
            self.DealInVar(v_name)
        elif self.state == 'loop' or self.state == 'if_branch':
            self.graph.without_out.add(e_node)
        return True

    # 自定义算子需要递归使用解析器，所以要使用一些特殊方法

    def CuOperator(self, query):
        """
        用于判断是否为自定义语句，开始自定义算子语句结构
        :param query:
        :return: True 合法语句，False 非法语句
        """
        c_o_reg = '(OPERATOR|operator)[ \t]+([a-zA-Z_]+[a-zA-Z0-9_]*)[ \t]*[(](.+)[)]{\n$'
        para_reg = '([a-zA-Z_]+[a-zA-Z0-9_]*[ \t]*,[ \t]*)*[a-zA-Z_]+[a-zA-Z0-9_]*'
        match_obj = re.match(c_o_reg, query)
        if match_obj:
            self.isCu = True
            if match_obj.group(2) is not None:
                self.operator = match_obj.group(2)
                parameter_str = re.search(para_reg, match_obj.group(3)).group()
            else:
                raise Exception('自定义算子参数为空，语句为：' + query)
            parameter_li = parameter_str.replace(' ', '').split(',')
            root = self.graph.nodes.pop(0)
            self.graph.without_out.remove(root)
            self.node_id = -1
            for p in parameter_li:
                self.node_id += 1
                node = Nd.InstantiationClass(self.node_id, 'Var', self.branches, vars=p)
                self.graph.InsertNode(node)
                self.UpdateVarList(p, self.node_id)
                self.input.append([p, node])
            return True
        else:
            return False

    def Reset(self):
        self.var_dict = dict()
        self.graph = DG.Graph()
        self.node_id = 0
        self.root_id = 0
        self.state = ''
        self.state_stack = list()
        self.loop_or_if_id = 0
        self.loop_id = 0
        self.out_var = dict()
        self.oth_branch = 0
        self.extra_pop_num = 0
        self.branches = list()
        self.input = list()
        self.operator = ''
        self.isCu = False
        root = Nd.InstantiationClass(self.node_id, 'Root', self.branches)
        self.graph.InsertNode(root)
        self.branches.append(0)

    @staticmethod
    def AddUserOperator(output, input, graph, operator):
        if os.path.exists('UserOperatorName.json'):
            with open('UserOperatorName.json', 'r') as f1:
                load_dict = json.load(f1)
            if operator not in load_dict.get('name'):
                load_dict.get('name').append(operator)
            with open('UserOperatorName.json', 'w') as f2:
                json.dump(load_dict, f2)
            with open('UserOperatorInfo', 'rb') as f3:
                data = pickle.load(f3)
            with open('UserOperatorInfo', 'wb+') as f4:
                data[operator] = [output, input, graph]
                pickle.dump(data, f4)
        else:
            with open('UserOperatorName.json', 'w') as f1:
                json.dump({'name': [operator]}, f1)
                with open('UserOperatorInfo', 'wb+') as f2:
                    data = {operator: [output, input, graph]}
                    pickle.dump(data, f2)


if __name__ == '__main__':
    from time import time
    with open('operators/LogitBoost.sql', 'r', encoding='utf-8') as f:
        create_test = f.readlines()
    testPar = Parser(create_test)
    result = testPar()
    # lp = LineProfiler()
    # lp.add_function()
    # executor = Executor(result)
    # s = time()
    # executor.run()
    # print(f'time:{time()-s} s')
    # print(executor.var_dict['loss'])
