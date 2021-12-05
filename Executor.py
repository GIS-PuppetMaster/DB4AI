from Nodes import LoopEnd
from Nodes import *
from utils import *
import numpy as np
import yaml
from collections import defaultdict
import torch
from copy import deepcopy
from gdbc import GDBC


class Executor:
    def __init__(self, graph):
        # with open('./config.yaml', encoding='utf-8') as f:
        #     config = yaml.load_all(f)
        # self.config = config
        self.raw_graph = graph
        self.var_dict = dict()
        self.tensor_dict = dict()
        self.backward_end = 0
        self.finished_loop_id = set()
        self.last_use = {}
        self.var_shape = {}
        # self.parameter_set = set()
        self.parameters = {}
        self.wait_to_be_release_after_loop = defaultdict(set)
        self.graph = graph
        self.cursor = GDBC()
        self.cursor.connect()
        # 有错误
        # self.graph = deepcopy(graph)
        # self.remove_extra_edges()
        self.init_executor()


    def remove_extra_edges(self):
        queue = []
        visited = set()
        root = self.graph.nodes[0]
        queue.append((root, None))
        while not len(queue) == 0:
            current_node, last_node = queue.pop(0)
            if last_node not in current_node.visited_sequence:
                current_node.visited_sequence.append(last_node)
            next_nodes = list(set([edge.end for edge in current_node.out_edges]))
            for node in next_nodes:
                if node not in visited:
                    queue.append((node, current_node))
                    if isinstance(node, Loop) or isinstance(node, LoopEnd):
                        visited.add(node)
        edges_to_removed = []
        for node in self.graph.nodes:
            start_nodes_of_edges_to_remove = node.visited_sequence[:-1]
            # 找出执行时需要移除的边
            edges_to_removed.extend(list(filter(lambda x: x.start in start_nodes_of_edges_to_remove, node.in_edges)))

        # 在start和end节点内移除
        def is_control_flow(x):
            return isinstance(x.start, If) or isinstance(x.start, IfBranch) or isinstance(x.end, IfEnd) or isinstance(
                x.end, LoopEnd) or isinstance(x.start, Loop)

        for node in self.graph.nodes:
            node.in_edges = list(
                filter(lambda x: x.start not in start_nodes_of_edges_to_remove or is_control_flow(x), node.in_edges))
            node.out_edges = list(
                filter(lambda x: x.end not in start_nodes_of_edges_to_remove or is_control_flow(x), node.out_edges))
        # 在图内移除
        self.graph.edges = list(filter(lambda x: x not in edges_to_removed or is_control_flow(x), self.graph.edges))
        # self.graph.Show()

    def init_executor(self):
        # for _, para in self.parameters:
        #     self.parameter_set.add(para)

        self.init_nodes()
        for var_name, node in self.last_use.items():
            if isinstance(node, If):
                for edge in node.out_edges:
                    edge.end.release_list.append(var_name)
            else:
                node.release_list.append(var_name)
        '''for node in self.graph.nodes:
            for var in node.vars:
                node.vars[node.vars.index(var)] = 'p' + var'''

    @bfs(True)
    def init_nodes(self, current_node, **kwargs):
        current_node.cursor = self.cursor
        current_node.cursor.connect()
        # if isinstance(current_node, Loop):
        #     if 'loop' not in kwargs['info'].keys():
        #         kwargs['info']['loop'] = {}
        #     loop_info = kwargs['info']['loop']
        #     loop_info[current_node.loop_id] = current_node
        # elif isinstance(current_node, LoopEnd):
        #     loop = kwargs['info']['loop'][current_node.loop_id]
        #     current_node.loop_pair = loop
        #     loop.loop_pair = current_node
        #     if 'break' in kwargs['info']:
        #         kwargs['info']['break'][current_node.loop_id].loop_pair = current_node
        # elif isinstance(current_node, Break):
        #     if 'break' not in kwargs['info'].keys():
        #         kwargs['info']['break'] = {}
        #     kwargs['info']['break'][current_node.loop_id] = current_node
        current_node.fathers = list(set([edge.start for edge in current_node.in_edges]))
        current_node.sons = list(set([edge.end for edge in current_node.out_edges]))
        current_node.branches_set = set(current_node.branches)
        current_node.infer_data()
        current_node.executor = self
        in_loop = -1
        for branch in current_node.branches:
            # 处于loop内部
            if isinstance(self.graph.nodes[branch], Loop):
                in_loop = branch
                break
        current_node.in_loop = in_loop
        # loop end
        if isinstance(current_node, LoopEnd):
            current_node.in_loop = current_node.loop_id
        if isinstance(current_node, If):
            for var_name, _ in current_node.out_edges[0].need_var:
                self.last_use[var_name] = current_node
        elif isinstance(current_node, Loop):
            if isinstance(current_node.dead_cycle, str):
                self.last_use[current_node.dead_cycle] = current_node
        elif isinstance(current_node, SaveParameters):
            for var_name in self.parameters[current_node.set_name]:
                self.last_use[var_name] = current_node
        else:
            for var_name in current_node.vars[1:]:
                self.last_use[var_name] = current_node

        return True

    def init_branches(self, node, current_branch):
        if isinstance(node, IfBranch):
            current_branch = node.id
        elif not (
                isinstance(node, If) or isinstance(node, IfEnd) or isinstance(node, Loop) or isinstance(node, LoopEnd)):
            node.branch = current_branch
        next_nodes = node.next_nodes()
        if len(next_nodes) == 0:
            return
        else:
            for next_node in next_nodes:
                self.init_branches(next_node, current_branch)

    @bfs(False)
    def execute(self, current_node, **kwargs):
        visited = kwargs['visited']
        # 确保父节点都执行完了再执行他
        if isinstance(current_node, LoopEnd) and current_node.return_next:
            pass
        else:
            for father in current_node.fathers:
                if not father.finished and not isinstance(father, LoopEnd):
                    if not isinstance(current_node, IfEnd):
                        return False
                    elif father.branches[-1] == current_node.selected_branch:
                        return False

        print(f'{current_node.id}')
        current_node.run(visited=visited, executor=self)
        # TODO: 遍历TENSORS，检查生成他的时候的作用域与当前是否相符，并且检查引用计数是否为0
        # for var_name in current_node.release_list:
        #     if var_name in self.var_dict.keys() and self.var_dict[var_name] is not None and not self.var_dict[var_name].requires_grad:
        #         if current_node.in_loop == -1 or re.match(r'^_[^_]+',var_name) is None:
        #             self.var_dict.pop(var_name)
        #         else:
        #             # loop结束后再回收
        #             self.wait_to_be_release_after_loop[current_node.in_loop].add(var_name)
        # # for var_name in list(self.var_dict.keys()):
        # #     if var_name not in self.last_use.keys():
        # #         self.var_dict.pop(var_name)
        # if isinstance(current_node, LoopEnd) and current_node.loop_id in self.finished_loop_id:
        #     for var_name in self.wait_to_be_release_after_loop[current_node.loop_id]:
        #         self.var_dict.pop(var_name)
        current_node.finished = True
        return True

    def run(self):
        self.execute()
