from Nodes import LoopEnd
from Nodes import *
from utils import *
import numpy as np
import yaml


class Executor:
    def __init__(self, graph):
        with open('./config.yaml', encoding='utf-8') as f:
            config = yaml.load_all(f)
        self.config = config
        self.graph = graph
        self.var_dict = dict()
        self.finished_loop_id = set()
        self.finished_nodes = set()
        self.last_use = {}
        self.init_executor()

    def init_executor(self):
        self.init_nodes()
        for var_name, node in self.last_use.items():
            node.release_list.append(var_name)

    @bfs(True)
    def init_nodes(self, current_node, **kwargs):
        if isinstance(current_node, Loop):
            if 'loop' not in kwargs['info'].keys():
                kwargs['info']['loop'] = {}
            loop_info = kwargs['info']['loop']
            loop_info[current_node.loop_id] = current_node
        elif isinstance(current_node, LoopEnd):
            loop = kwargs['info']['loop'][current_node.loop_id]
            current_node.loop_pair = loop
            loop.loop_pair = current_node
            if 'break' in kwargs['info']:
                kwargs['info']['break'][current_node.loop_id].loop_pair = current_node
        elif isinstance(current_node, Break):
            if 'break' not in kwargs['info'].keys():
                kwargs['info']['break'] = {}
            kwargs['info']['break'][current_node.loop_id] = current_node
        current_node.fathers = [edge.start for edge in current_node.in_edges]
        current_node.sons = [edge.end for edge in current_node.out_edges]
        current_node.branches_set = set(current_node.branches)
        current_node.infer_data()
        current_node.executor = self
        # TODO: 执行loop内部时不回收，只回收带@的
        for var_name in current_node.vars[1:]:
            self.last_use[var_name] = current_node
        return True

    def init_branches(self, node, current_branch):
        if isinstance(node, IfBranch):
            current_branch = node.id
        elif not (isinstance(node, If) or isinstance(node, IfEnd) or isinstance(node, Loop) or isinstance(node, LoopEnd)):
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
        if not isinstance(current_node, IfEnd) and not isinstance(current_node, LoopEnd):
            for father in current_node.fathers:
                if father not in self.finished_nodes and not isinstance(father, LoopEnd):
                    return False
        current_node.run(visited=visited, executor=self)
        # TODO: 对requires_grad对象的回收
        for var_name in current_node.release_list:
            if not self.var_dict[var_name].requires_grad:
                self.var_dict.pop(var_name)
        for var_name in list(self.var_dict.keys()):
            if var_name not in self.last_use.keys() or ('@' in var_name and not self.var_dict[var_name].requires_grad):
                self.var_dict.pop(var_name)
        self.finished_nodes.add(current_node)
        return True

    def run(self):
        self.execute()
